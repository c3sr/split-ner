import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.bert_mrc import BERT_MRC
from src.components.crf import CRF
from src.models.base import BaseExecutor
from src.models.type_cnn_lstm_crf import TypeCRFDataset
from src.utils.evaluator import Evaluator
from src.utils.general import set_all_seeds, parse_config


class BERT_MRC_CRF(nn.Module):

    def __init__(self, out_tag_names, out_tags, hidden_dim, out_dim, device, use_word=True, post_padding=True,
                 pad_tag="<PAD>", word_emb_model_from_tf=False):
        super(BERT_MRC_CRF, self).__init__()

        self.bert_mrc = BERT_MRC(out_tag_names=out_tag_names, out_dim=out_dim, hidden_dim=hidden_dim,
                                 use_word=use_word, device=device, word_emb_model_from_tf=word_emb_model_from_tf)
        self.crf = CRF(out_tags=out_tags, device=device, post_padding=post_padding, pad_tag=pad_tag)

    def neg_log_likelihood(self, text, word_x, char_x, type_x, word_mask, char_mask, tags):
        feats = self.bert_mrc(text, word_x, char_x, type_x, word_mask, char_mask)
        forward_score = self.crf.forward_algo(feats, word_mask)
        gold_score = self.crf.score_sentence(feats, word_mask, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, text, word_x, char_x, type_x, word_mask, char_mask):
        feats = self.bert_mrc(text, word_x, char_x, type_x, word_mask, char_mask)
        score, tag_seq = self.crf.veterbi_decode(feats, word_mask)
        return score, tag_seq


class BERT_MRC_CRFExecutor(BaseExecutor):

    def __init__(self, config):
        super(BERT_MRC_CRFExecutor, self).__init__(config)

        self.config.data.inp_tag_vocab_path = os.path.join(self.config.data.data_dir,
                                                           self.config.data.inp_tag_vocab_path)
        self.config.data.tag_emb_path = os.path.join(self.config.data.data_dir, self.config.data.tag_emb_path)

        self.define_datasets()

        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=config.batch_size,
                                            shuffle=self.shuffle_train_data)
        self.dev_data_loader = DataLoader(dataset=self.dev_dataset, batch_size=config.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=config.batch_size, shuffle=False)

        self.model = BERT_MRC_CRF(out_tags=self.get_model_training_out_tags(),
                                  out_dim=self.get_model_training_out_dim(), hidden_dim=self.config.hidden_dim,
                                  device=self.device, use_word=self.config.use_word, pad_tag=self.config.pad_tag,
                                  post_padding=self.config.post_padding,
                                  word_emb_model_from_tf=self.config.word_emb_model_from_tf,
                                  out_tag_names=self.train_dataset.out_tag_names)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=config.lr)

    def define_datasets(self):
        self.train_dataset = TypeCRFDataset(config=self.config, corpus_path=self.config.data.train_path)
        self.dev_dataset = TypeCRFDataset(config=self.config, corpus_path=self.config.data.dev_path)
        self.test_dataset = TypeCRFDataset(config=self.config, corpus_path=self.config.data.test_path)

    def get_model_training_out_dim(self):
        return len(self.train_dataset.out_tags)

    def get_model_training_out_tags(self):
        return self.train_dataset.out_tags

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        with tqdm(self.train_data_loader) as progress_bar:
            for text, word_feature, char_feature, type_feature, word_mask, char_mask, label in progress_bar:
                text = np.array(text).T.tolist()
                word_feature = word_feature.to(self.device)
                char_feature = char_feature.to(self.device)
                type_feature = type_feature.to(self.device)
                word_mask = word_mask.to(self.device)
                char_mask = char_mask.to(self.device)
                label = label.to(self.device)
                batch_size = label.shape[0]
                self.optimizer.zero_grad()
                loss = self.model.neg_log_likelihood(text, word_feature, char_feature, type_feature, word_mask,
                                                     char_mask,
                                                     label)
                progress_bar.set_postfix(Epoch=epoch, Batch_Loss="{0:.3f}".format(loss.item() / batch_size))
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
        print("TRAIN: Epoch: {0} | Loss:{1:.3f}".format(epoch, train_loss / len(self.train_data_loader.dataset)))

    def evaluate_epoch(self, data_loader, epoch, prefix, outfile=None):
        self.model.eval()
        total_score = 0.0
        total_text = []
        total_prediction = []
        total_label = []
        for text, word_feature, char_feature, type_feature, word_mask, char_mask, label in data_loader:
            text = np.array(text).T.tolist()
            word_feature = word_feature.to(self.device)
            char_feature = char_feature.to(self.device)
            type_feature = type_feature.to(self.device)
            word_mask = word_mask.to(self.device)
            char_mask = char_mask.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                score, prediction = self.model(text, word_feature, char_feature, type_feature, word_mask, char_mask)
                total_text.extend(text)
                total_prediction.append(prediction)
                total_label.append(label.clone())
                total_score += score.sum().item()
        total_prediction = np.vstack([np.array(p) for p in total_prediction])
        total_label = torch.cat(total_label, dim=0).cpu().numpy()
        if outfile:
            self.print_outputs(corpus=total_text, gold=total_label, predicted=total_prediction,
                               mapping=self.get_model_training_out_tags(), outfile=outfile)
        evaluator = Evaluator(gold=total_label, predicted=total_prediction, tags=self.get_model_training_out_tags(),
                              ignore_tags=[self.config.none_tag, self.config.pad_tag], none_tag=self.config.none_tag,
                              pad_tag=self.config.pad_tag)
        mean_score = total_score / len(data_loader.dataset)
        print("{0}: Epoch: {1} | Token-Level Micro F1: {2:.3f} | Score: {3:.3f}".format(
            prefix, epoch, evaluator.significant_token_metric.micro_avg_f1(), mean_score))
        if self.config.verbose:
            print("Entity-Level Metrics:")
            print(evaluator.entity_metric.report())
            print("Token-Level Metrics:")
            print(evaluator.significant_token_metric.report())

        return mean_score, evaluator

    def query(self, sentence_text):
        self.model.eval()

        sentence_tokens = BERT_MRC_CRFExecutor.get_query_tokens(sentence_text)
        text, word_feature, char_feature, type_feature, word_mask, char_mask, _ = self.test_dataset.get_query_given_tokens(
            sentence_tokens)
        text = [text]
        word_feature = torch.as_tensor(word_feature, device=self.device).unsqueeze(0)
        char_feature = torch.as_tensor(char_feature, device=self.device).unsqueeze(0)
        type_feature = torch.as_tensor(type_feature, device=self.device).unsqueeze(0)
        word_mask = torch.as_tensor(word_mask, device=self.device).unsqueeze(0)
        char_mask = torch.as_tensor(char_mask, device=self.device).unsqueeze(0)
        with torch.no_grad():
            score, prediction = self.model(text, word_feature, char_feature, type_feature, word_mask, char_mask)
        prediction = prediction[0]
        for i in range(len(prediction)):
            print("{0}\t{1}".format(text[0][i], self.test_dataset.out_tags[prediction[i]]))


def main(args):
    config = parse_config(args.config)
    set_all_seeds(config.seed)
    executor = BERT_MRC_CRFExecutor(config)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BERT-MRC-CRF Model for Sequence Labeling")
    ap.add_argument("--config", default="../configs/config.json", help="config file")
    ap = ap.parse_args()
    main(ap)
