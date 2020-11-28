import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.cnn_lstm import CNN_LSTM
from src.components.crf import CRF
from src.models.base import BaseExecutor
from src.reader.type_dataset import TypeDataset
from src.utils.evaluator import Evaluator
from src.utils.general import set_all_seeds, parse_config


class CNN_LSTM_CRF(nn.Module):

    def __init__(self, out_tags, inp_dim, conv1_dim, hidden_dim, out_dim, kernel_size, word_len, device,
                 word_vocab_size=None, pos_tag_vocab_size=None, dep_tag_vocab_size=None, word_emb_dim=None,
                 pos_tag_emb_dim=None, dep_tag_emb_dim=None, tag_emb_dim=None, pre_trained_emb=None, use_word=True,
                 use_char=True, use_maxpool=False, use_pos_tag=False, use_dep_tag=False, use_lstm=False,
                 use_tag_info="self", post_padding=True, dropout_ratio=0.5,
                 pad_tag="<PAD>", use_tag_cosine_sim=False, fine_tune_bert=False, use_tfo="none",
                 use_class_guidance=False, tag_emb=None, word_emb_model_from_tf=False, num_lstm_layers=1):
        super(CNN_LSTM_CRF, self).__init__()

        self.cnn_lstm = CNN_LSTM(inp_dim=inp_dim, conv1_dim=conv1_dim, out_dim=out_dim, hidden_dim=hidden_dim,
                                 kernel_size=kernel_size, word_len=word_len, word_vocab_size=word_vocab_size,
                                 pos_tag_vocab_size=pos_tag_vocab_size,
                                 dep_tag_vocab_size=dep_tag_vocab_size,
                                 tag_emb_dim=tag_emb_dim, pos_tag_emb_dim=pos_tag_emb_dim,
                                 dep_tag_emb_dim=dep_tag_emb_dim, use_lstm=use_lstm,
                                 word_emb_dim=word_emb_dim, dropout_ratio=dropout_ratio,
                                 pre_trained_emb=pre_trained_emb, use_char=use_char, use_word=use_word,
                                 use_pos_tag=use_pos_tag, use_dep_tag=use_dep_tag,
                                 use_maxpool=use_maxpool, use_tag_info=use_tag_info, device=device,
                                 use_tag_cosine_sim=use_tag_cosine_sim, fine_tune_bert=fine_tune_bert, use_tfo=use_tfo,
                                 use_class_guidance=use_class_guidance, tag_emb=tag_emb,
                                 word_emb_model_from_tf=word_emb_model_from_tf, num_lstm_layers=num_lstm_layers)

        self.crf = CRF(out_tags=out_tags, device=device, post_padding=post_padding, pad_tag=pad_tag)

    def neg_log_likelihood(self, text, word_x, char_x, type_x, word_mask, char_mask, tags):
        feats = self.cnn_lstm(text, word_x, char_x, type_x, word_mask, char_mask)
        forward_score = self.crf.forward_algo(feats, word_mask)
        gold_score = self.crf.score_sentence(feats, word_mask, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, text, word_x, char_x, type_x, word_mask, char_mask):
        feats = self.cnn_lstm(text, word_x, char_x, type_x, word_mask, char_mask)
        score, tag_seq = self.crf.veterbi_decode(feats, word_mask)
        return score, tag_seq


class TypeCRFDataset(TypeDataset):

    def parse_tags(self):
        TypeDataset.parse_tags(self)
        self.out_tags.append(CRF.START_TAG)
        self.out_tags.append(CRF.STOP_TAG)

    def read_tag_emb(self, tags, emb_dict, tag_emb_dim):
        tag_emb = []
        for tag in tags:
            if tag.startswith("B-") or tag.startswith("I-"):
                root = tag[2:]
                root_vec = emb_dict[root] if root in emb_dict else [0.0] * tag_emb_dim  # send 0's if not found
                bi_vec = [0.0] if tag.startswith("B-") else [1.0]
                tag_emb.append(root_vec + bi_vec + [0.0, 0.0])
            else:
                # special tags
                if tag not in emb_dict:
                    main_vec = [0.0] * tag_emb_dim + [0.0]
                else:
                    main_vec = emb_dict[tag] + [0.0]

                if tag == self.config.none_tag:
                    tag_emb.append(main_vec + [1.0, 0.0])
                elif tag == self.config.pad_tag:
                    tag_emb.append(main_vec + [0.0, 1.0])
                elif tag == CRF.START_TAG or tag == CRF.STOP_TAG:  # not used, but embedded to avoid errors
                    tag_emb.append(main_vec + [0.0, 0.0])
                else:
                    raise ValueError("unexpected tag: {0}".format(tag))
        return tag_emb


class TypeCNN_LSTM_CRFExecutor(BaseExecutor):

    def __init__(self, config):
        super(TypeCNN_LSTM_CRFExecutor, self).__init__(config)

        train_char_emb = self.config.use_char != "none" or self.config.pattern.use_pattern != "none"
        use_lstm = not self.config.no_lstm

        # TODO: DEBUG LOGGING
        # self.logger = ConfigParser().get_logger('trainer', config['trainer']['verbosity'])
        # self.logger = ConfigParser().get_logger('trainer')
        # # self.writer = WriterTensorboardX("temp_dir", self.logger, cfg_trainer['tensorboardX'])
        # self.writer = WriterTensorboardX("temp_dir", self.logger, True)
        #
        # self.logger.info('    {:15s}: {}'.format(str("key"), 10.0))
        # self.writer.add_text('Text', 'Model Architecture: {}'.format("abcd"), 0)
        # self.writer.add_scalar('{}'.format("accuracy"), 93.56)

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

        self.model = CNN_LSTM_CRF(out_tags=self.get_model_training_out_tags(), inp_dim=self.train_dataset.inp_dim,
                                  conv1_dim=self.config.conv1_dim, out_dim=self.get_model_training_out_dim(),
                                  hidden_dim=self.config.hidden_dim, kernel_size=config.kernel_size,
                                  word_len=self.config.max_word_len, device=self.device,
                                  word_vocab_size=len(self.train_dataset.word_vocab),
                                  pos_tag_vocab_size=len(self.train_dataset.pos_tag_vocab),
                                  dep_tag_vocab_size=len(self.train_dataset.dep_tag_vocab), use_lstm=use_lstm,
                                  word_emb_dim=self.config.word_emb_dim,
                                  tag_emb_dim=self.train_dataset.tag_emb_dim,
                                  pos_tag_emb_dim=self.config.pos_tag_emb_dim,
                                  dep_tag_emb_dim=self.config.dep_tag_emb_dim, pre_trained_emb=pre_trained_emb,
                                  use_char=train_char_emb, use_word=self.config.use_word,
                                  use_maxpool=self.config.use_maxpool, use_pos_tag=self.config.use_pos_tag,
                                  use_dep_tag=self.config.use_dep_tag,
                                  use_tag_info=self.config.use_tag_info, pad_tag=self.config.pad_tag,
                                  post_padding=self.config.post_padding,
                                  use_tag_cosine_sim=self.config.use_tag_cosine_sim,
                                  fine_tune_bert=self.config.fine_tune_bert, use_tfo=self.config.use_tfo,
                                  use_class_guidance=self.config.use_class_guidance, tag_emb=tag_emb,
                                  word_emb_model_from_tf=self.config.word_emb_model_from_tf,
                                  num_lstm_layers=self.config.num_lstm_layers, dropout_ratio=self.config.dropout_ratio)

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

        sentence_tokens = TypeCNN_LSTM_CRFExecutor.get_query_tokens(sentence_text)
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
    executor = TypeCNN_LSTM_CRFExecutor(config)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CNN-LSTM-CRF Model for Sequence Labeling")
    ap.add_argument("--config", default="../configs/config.json", help="config file")
    ap = ap.parse_args()
    main(ap)
