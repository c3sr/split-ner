import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.cnn_lstm import CNN_LSTM
from src.models.base import BaseExecutor
from src.reader.type_dataset import TypeDataset
from src.utils.evaluator import Evaluator
from src.utils.general import set_all_seeds, parse_config


class TypeCNN_LSTMExecutor(BaseExecutor):

    def __init__(self, config):
        super(TypeCNN_LSTMExecutor, self).__init__(config)

        self.config.data.inp_tag_vocab_path = os.path.join(self.config.data.data_dir,
                                                           self.config.data.inp_tag_vocab_path)
        self.config.data.tag_emb_path = os.path.join(self.config.data.data_dir, self.config.data.tag_emb_path)

        self.define_datasets()

        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=config.batch_size,
                                            shuffle=self.shuffle_train_data)
        self.dev_data_loader = DataLoader(dataset=self.dev_dataset, batch_size=config.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=config.batch_size, shuffle=False)

        pre_trained_emb = None
        if self.config.use_word == "glove":
            pre_trained_emb = torch.as_tensor(self.train_dataset.word_emb, device=self.device)

        tag_emb = None
        if self.config.use_tag_cosine_sim or self.config.use_class_guidance:
            tag_emb = self.prep_tag_emb_tensor()

        train_char_emb = self.config.use_char != "none" or self.config.pattern.use_pattern != "none"
        use_lstm = not self.config.no_lstm

        self.model = CNN_LSTM(inp_dim=self.train_dataset.inp_dim, conv1_dim=self.config.conv1_dim,
                              out_dim=self.get_model_training_out_dim(), hidden_dim=self.config.hidden_dim,
                              kernel_size=config.kernel_size, word_len=self.config.max_word_len,
                              word_vocab_size=len(self.train_dataset.word_vocab),
                              pos_tag_vocab_size=len(self.train_dataset.pos_tag_vocab),
                              dep_tag_vocab_size=len(self.train_dataset.dep_tag_vocab), use_lstm=use_lstm,
                              word_emb_dim=self.config.word_emb_dim, dropout_ratio=self.config.dropout_ratio,
                              tag_emb_dim=self.train_dataset.tag_emb_dim, pos_tag_emb_dim=self.config.pos_tag_emb_dim,
                              dep_tag_emb_dim=self.config.dep_tag_emb_dim, pre_trained_emb=pre_trained_emb,
                              use_char=train_char_emb, use_word=self.config.use_word,
                              use_maxpool=self.config.use_maxpool,
                              use_pos_tag=self.config.use_pos_tag, use_dep_tag=self.config.use_dep_tag,
                              use_tag_info=self.config.use_tag_info, device=self.device,
                              use_tag_cosine_sim=self.config.use_tag_cosine_sim,
                              fine_tune_bert=self.config.fine_tune_bert, use_tfo=self.config.use_tfo,
                              use_class_guidance=self.config.use_class_guidance, tag_emb=tag_emb,
                              word_emb_model_from_tf=self.config.word_emb_model_from_tf,
                              num_lstm_layers=self.config.num_lstm_layers)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=config.lr)

    def define_datasets(self):
        self.train_dataset = TypeDataset(config=self.config, corpus_path=self.config.data.train_path)
        self.dev_dataset = TypeDataset(config=self.config, corpus_path=self.config.data.dev_path)
        self.test_dataset = TypeDataset(config=self.config, corpus_path=self.config.data.test_path)

    def get_model_training_out_dim(self):
        return len(self.train_dataset.out_tags)

    def get_model_training_out_tags(self):
        return self.train_dataset.out_tags

    def prep_tag_emb_tensor(self):
        model_training_out_tags = self.get_model_training_out_tags()
        tag_emb = []
        for tag in model_training_out_tags:
            index = self.train_dataset.out_tags.index(tag)
            tag_emb.append(self.train_dataset.out_tag_emb[index])
        return torch.as_tensor(np.array(tag_emb), device=self.device)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_prediction = []
        train_label = []
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
                prediction = self.model(text, word_feature, char_feature, type_feature, word_mask, char_mask)
                train_prediction.append(prediction.detach().clone())
                train_label.append(label.detach().clone())
                prediction = prediction.transpose(2, 1)
                loss = self.criterion(prediction, label)
                progress_bar.set_postfix(Epoch=epoch, Batch_Loss="{0:.3f}".format(loss.item() / batch_size))
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

        train_prediction = torch.cat(train_prediction, dim=0)
        train_prediction = torch.argmax(train_prediction, dim=2)

        train_prediction = train_prediction.cpu().numpy()
        train_label = torch.cat(train_label, dim=0).cpu().numpy()

        evaluator = Evaluator(gold=train_label, predicted=train_prediction, tags=self.train_dataset.out_tags,
                              ignore_tags=[self.config.none_tag, self.config.pad_tag], none_tag=self.config.none_tag,
                              pad_tag=self.config.pad_tag)
        print("TRAIN: Epoch: {0} | Loss:{1:.3f} | Token-Level Micro F1: {2:.3f}".format(epoch, train_loss / len(
            self.train_data_loader.dataset), evaluator.significant_token_metric.micro_avg_f1()))

    def evaluate_epoch(self, data_loader, epoch, prefix, outfile=None):
        self.model.eval()
        total_loss = 0.0
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
                prediction = self.model(text, word_feature, char_feature, type_feature, word_mask, char_mask)
                total_text.extend(text)
                total_prediction.append(prediction.clone())
                total_label.append(label.clone())
                prediction = prediction.transpose(2, 1)
                loss = self.criterion(prediction, label)
                total_loss += loss.item()
        total_prediction = torch.cat(total_prediction, dim=0)
        total_prediction = torch.argmax(total_prediction, dim=2)

        total_prediction = total_prediction.cpu().numpy()
        total_label = torch.cat(total_label, dim=0).cpu().numpy()

        if outfile:
            self.print_outputs(corpus=total_text, gold=total_label, predicted=total_prediction,
                               mapping=data_loader.dataset.out_tags, outfile=outfile)
        evaluator = Evaluator(gold=total_label, predicted=total_prediction, tags=data_loader.dataset.out_tags,
                              ignore_tags=[self.config.none_tag, self.config.pad_tag], none_tag=self.config.none_tag,
                              pad_tag=self.config.pad_tag)
        mean_loss = total_loss / len(data_loader.dataset)
        print("{0}: Epoch: {1} | Token-Level Micro F1: {2:.3f} | Loss: {3:.3f}".format(
            prefix, epoch, evaluator.significant_token_metric.micro_avg_f1(), mean_loss))
        if self.config.verbose:
            print("Entity-Level Metrics:")
            print(evaluator.entity_metric.report())
            print("Token-Level Metrics:")
            print(evaluator.significant_token_metric.report())

        return mean_loss, evaluator

    def query(self, sentence_text):
        self.model.eval()

        sentence_tokens = TypeCNN_LSTMExecutor.get_query_tokens(sentence_text)
        text, word_feature, char_feature, type_feature, word_mask, char_mask, _ = self.test_dataset.get_query_given_tokens(
            sentence_tokens)
        text = [text]
        word_feature = torch.as_tensor(word_feature, device=self.device).unsqueeze(0)
        char_feature = torch.as_tensor(char_feature, device=self.device).unsqueeze(0)
        type_feature = torch.as_tensor(type_feature, device=self.device).unsqueeze(0)
        word_mask = torch.as_tensor(word_mask, device=self.device).unsqueeze(0)
        char_mask = torch.as_tensor(char_mask, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(text, word_feature, char_feature, type_feature, word_mask, char_mask)
        prediction = np.argmax(prediction, axis=2).squeeze(0)
        for i in range(prediction.shape[0]):
            print("{0}\t{1}".format(text[0][i], self.test_dataset.out_tags[prediction[i]]))


def main(args):
    config = parse_config(args.config)
    set_all_seeds(config.seed)
    executor = TypeCNN_LSTMExecutor(config)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CNN-LSTM Model for Sequence Labeling")
    ap.add_argument("--config", default="../configs/config.json", help="config file")
    ap = ap.parse_args()
    main(ap)
