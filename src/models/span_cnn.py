import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.cnn_lstm import CNN_LSTM_Span
from src.models.base import BaseExecutor
from src.reader.span_dataset import SpanTypeDataset
from src.utils.evaluator import SpanEvaluator, Span
from src.utils.general import set_all_seeds, parse_config


class SpanLoss(nn.Module):

    def __init__(self, loss_weight_begin=1.0, loss_weight_end=1.0, loss_weight_span=1.0, masking="pre"):
        super(SpanLoss, self).__init__()

        self.masking = masking

        self.boundary_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.span_loss_fn = nn.BCEWithLogitsLoss(reduction="none")  # for nested classification

        self.loss_wb = loss_weight_begin
        self.loss_we = loss_weight_end
        self.loss_ws = loss_weight_span

    def forward(self, begin_x, end_x, span_x, word_mask, begin_positions=None, end_positions=None, span_positions=None):
        if begin_positions is not None and end_positions is not None:

            if self.masking == "pre":
                begin_x = begin_x * word_mask.unsqueeze(-1).expand_as(begin_x)
                end_x = end_x * word_mask.unsqueeze(-1).expand_as(end_x)
                # logic: span_mask[b][i][j][t] = 0 for all t, if word_mask[b][i] = 0 or word_mask[b][j] = 0
                span_mask = torch.matmul(word_mask.unsqueeze(-1).float(), word_mask.unsqueeze(1).float()).unsqueeze(
                    -1).expand_as(span_x)
                span_x = span_x * span_mask

            begin_loss = self.boundary_loss_fn(begin_x.view(-1, 2), begin_positions.view(-1))
            end_loss = self.boundary_loss_fn(end_x.view(-1, 2), end_positions.view(-1))
            span_loss = self.span_loss_fn(span_x, span_positions.float())

            if self.masking == "post":
                begin_loss = begin_loss * word_mask.view(-1)
                end_loss = end_loss * word_mask.view(-1)
                # logic: span_mask[b][i][j][t] = 0 for all t, if word_mask[b][i] = 0 or word_mask[b][j] = 0
                span_mask = torch.matmul(word_mask.unsqueeze(-1).float(), word_mask.unsqueeze(1).float()).unsqueeze(
                    -1).expand_as(span_loss)
                span_loss = span_loss * span_mask

            begin_loss = torch.mean(begin_loss)
            end_loss = torch.mean(end_loss)
            span_loss = torch.mean(span_loss)

            total_loss = self.loss_wb * begin_loss + self.loss_we * end_loss + self.loss_ws * span_loss
            return total_loss

        # Evaluation
        begin_x = torch.argmax(begin_x, dim=-1)
        end_x = torch.argmax(end_x, dim=-1)
        span_x = torch.sigmoid(span_x)

        begin_x = begin_x * word_mask
        end_x = end_x * word_mask
        # logic: span_mask[b][i][j][t] = 0 for all t, if word_mask[b][i] = 0 or word_mask[b][j] = 0
        span_mask = torch.matmul(word_mask.unsqueeze(-1).float(), word_mask.unsqueeze(1).float()).unsqueeze(
            -1).expand_as(span_x)
        span_x = span_x * span_mask

        return begin_x, end_x, span_x


class SpanExecutor(BaseExecutor):
    SPAN_THRESHOLD = 0.5

    def __init__(self, config):
        super(SpanExecutor, self).__init__(config)

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

        self.model = CNN_LSTM_Span(inp_dim=self.train_dataset.inp_dim, conv1_dim=self.config.conv1_dim,
                                   out_dim=self.get_model_training_out_dim(), hidden_dim=self.config.hidden_dim,
                                   kernel_size=config.kernel_size, word_len=self.config.max_word_len,
                                   word_vocab_size=len(self.train_dataset.word_vocab),
                                   pos_tag_vocab_size=len(self.train_dataset.pos_tag_vocab),
                                   dep_tag_vocab_size=len(self.train_dataset.dep_tag_vocab), use_lstm=use_lstm,
                                   word_emb_dim=self.config.word_emb_dim,
                                   tag_emb_dim=self.train_dataset.tag_emb_dim,
                                   pos_tag_emb_dim=self.config.pos_tag_emb_dim,
                                   dep_tag_emb_dim=self.config.dep_tag_emb_dim, pre_trained_emb=pre_trained_emb,
                                   use_char=train_char_emb, use_word=self.config.use_word,
                                   use_maxpool=self.config.use_maxpool,
                                   use_pos_tag=self.config.use_pos_tag, use_dep_tag=self.config.use_dep_tag,
                                   use_tag_info=self.config.use_tag_info, device=self.device,
                                   use_tag_cosine_sim=self.config.use_tag_cosine_sim,
                                   fine_tune_bert=self.config.fine_tune_bert, use_tfo=self.config.use_tfo,
                                   use_class_guidance=self.config.use_class_guidance, tag_emb=tag_emb,
                                   span_pooling=self.config.span_pooling,
                                   word_emb_model_from_tf=self.config.word_emb_model_from_tf,
                                   num_lstm_layers=self.config.num_lstm_layers)

        self.criterion = SpanLoss(masking=self.config.loss_masking)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=config.lr)

    def define_datasets(self):
        self.train_dataset = SpanTypeDataset(corpus_path=self.config.data.train_path, config=self.config)
        self.dev_dataset = SpanTypeDataset(corpus_path=self.config.data.dev_path, config=self.config)
        self.test_dataset = SpanTypeDataset(corpus_path=self.config.data.test_path, config=self.config)

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
        total_loss = 0.0

        condensed_pred_span_position = []
        condensed_gold_span_position = []

        with tqdm(self.train_data_loader) as progress_bar:
            for text, \
                word_feature, \
                char_feature, \
                type_feature, \
                word_mask, \
                char_mask, \
                gold_begin_position, \
                gold_end_position, \
                gold_span_position \
                    in progress_bar:
                text = np.array(text).T.tolist()
                word_feature = word_feature.to(self.device)
                char_feature = char_feature.to(self.device)
                type_feature = type_feature.to(self.device)
                word_mask = word_mask.to(self.device)
                char_mask = char_mask.to(self.device)
                gold_begin_position = gold_begin_position.to(self.device)
                gold_end_position = gold_end_position.to(self.device)
                gold_span_position = gold_span_position.to(self.device)
                batch_size = len(text)
                self.optimizer.zero_grad()
                pred_begin_position, pred_end_position, pred_span_position = self.model(text, word_feature,
                                                                                        char_feature, type_feature,
                                                                                        word_mask, char_mask)

                loss = self.criterion(pred_begin_position, pred_end_position, pred_span_position, word_mask,
                                      gold_begin_position, gold_end_position, gold_span_position)
                progress_bar.set_postfix(Epoch=epoch, Batch_Loss="{0:.3f}".format(loss.item()))
                total_loss += loss.item() * batch_size
                loss.backward()
                self.optimizer.step()

                # for evaluation
                pred_begin_position, pred_end_position, pred_span_position = self.criterion(pred_begin_position,
                                                                                            pred_end_position,
                                                                                            pred_span_position,
                                                                                            word_mask)

                condensed_pred_span_position.extend(
                    SpanExecutor.prep_spans(begin_position=pred_begin_position.detach().cpu().clone().numpy(),
                                            end_position=pred_end_position.detach().cpu().clone().numpy(),
                                            span_position=pred_span_position.detach().cpu().clone().numpy(),
                                            tags=self.train_dataset.out_tags,
                                            threshold=SpanExecutor.SPAN_THRESHOLD))

                condensed_gold_span_position.extend(
                    SpanExecutor.prep_spans(begin_position=gold_begin_position.detach().cpu().clone().numpy(),
                                            end_position=gold_end_position.detach().cpu().clone().numpy(),
                                            span_position=gold_span_position.detach().cpu().clone().numpy(),
                                            tags=self.train_dataset.out_tags,
                                            threshold=SpanExecutor.SPAN_THRESHOLD))

        evaluator = SpanEvaluator(gold=condensed_gold_span_position, predicted=condensed_pred_span_position,
                                  tags=self.train_dataset.out_tags)
        print("TRAIN: Epoch: {0} | Loss:{1:.3f} | Entity-Span-Level Micro F1: {2:.3f}".format(epoch, total_loss / len(
            self.train_data_loader.dataset), evaluator.entity_metric.micro_avg_f1()))

    @staticmethod
    def prep_spans(begin_position, end_position, span_position, tags, threshold):
        spans = []
        for sent_begin_position, sent_end_position, sent_span_position in zip(begin_position, end_position,
                                                                              span_position):
            sent_spans = []
            begin_indices = [i for i, p in enumerate(sent_begin_position) if p != 0]
            end_indices = [i for i, p in enumerate(sent_end_position) if p != 0]

            for begin_index in begin_indices:
                candidate_end_indices = [i for i in end_indices if i >= begin_index]
                for candidate_end_index in candidate_end_indices:
                    for tag_index in range(len(tags)):
                        if sent_span_position[begin_index][candidate_end_index][tag_index] >= threshold:
                            sent_spans.append(Span(start=begin_index, end=candidate_end_index, tag=tags[tag_index]))
            spans.append(sent_spans)

        return spans

    def evaluate_epoch(self, data_loader, epoch, prefix, outfile=None):
        self.model.eval()
        total_loss = 0.0
        total_text = []

        condensed_pred_span_position = []
        condensed_gold_span_position = []

        with tqdm(data_loader) as progress_bar:
            for text, \
                word_feature, \
                char_feature, \
                type_feature, \
                word_mask, \
                char_mask, \
                gold_begin_position, \
                gold_end_position, \
                gold_span_position \
                    in progress_bar:
                text = np.array(text).T.tolist()
                word_feature = word_feature.to(self.device)
                char_feature = char_feature.to(self.device)
                type_feature = type_feature.to(self.device)
                word_mask = word_mask.to(self.device)
                char_mask = char_mask.to(self.device)
                gold_begin_position = gold_begin_position.to(self.device)
                gold_end_position = gold_end_position.to(self.device)
                gold_span_position = gold_span_position.to(self.device)
                batch_size = len(text)

                with torch.no_grad():
                    pred_begin_position, pred_end_position, pred_span_position = self.model(text, word_feature,
                                                                                            char_feature, type_feature,
                                                                                            word_mask, char_mask)

                    loss = self.criterion(pred_begin_position, pred_end_position, pred_span_position, word_mask,
                                          gold_begin_position, gold_end_position, gold_span_position)
                    progress_bar.set_postfix(Epoch=epoch, Batch_Loss="{0:.3f}".format(loss.item()))
                    total_loss += loss.item() * batch_size

                    pred_begin_position, pred_end_position, pred_span_position = self.criterion(pred_begin_position,
                                                                                                pred_end_position,
                                                                                                pred_span_position,
                                                                                                word_mask)
                    total_text.extend(text)
                    condensed_pred_span_position.extend(
                        SpanExecutor.prep_spans(begin_position=pred_begin_position.detach().cpu().clone().numpy(),
                                                end_position=pred_end_position.detach().cpu().clone().numpy(),
                                                span_position=pred_span_position.detach().cpu().clone().numpy(),
                                                tags=self.train_dataset.out_tags,
                                                threshold=SpanExecutor.SPAN_THRESHOLD))

                    condensed_gold_span_position.extend(
                        SpanExecutor.prep_spans(begin_position=gold_begin_position.detach().cpu().clone().numpy(),
                                                end_position=gold_end_position.detach().cpu().clone().numpy(),
                                                span_position=gold_span_position.detach().cpu().clone().numpy(),
                                                tags=self.train_dataset.out_tags,
                                                threshold=SpanExecutor.SPAN_THRESHOLD))

        if outfile:
            self.print_outputs(corpus=total_text, gold=condensed_gold_span_position,
                               predicted=condensed_pred_span_position,
                               mapping=data_loader.dataset.out_tags, outfile=outfile)
        evaluator = SpanEvaluator(gold=condensed_gold_span_position, predicted=condensed_pred_span_position,
                                  tags=self.train_dataset.out_tags)
        mean_loss = total_loss / len(data_loader.dataset)
        print("{0}: Epoch: {1} | Entity-Span-Level Micro F1: {2:.3f} | Loss: {3:.3f}".format(
            prefix, epoch, evaluator.entity_metric.micro_avg_f1(), mean_loss))
        if self.config.verbose:
            print("Entity-Level Metrics:")
            print(evaluator.entity_metric.report())

        return mean_loss, evaluator

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.config.num_epochs):
            self.train_epoch(epoch)
            self.save_checkpoint(epoch, save_best=False)

            _, dev_evaluator = self.evaluate_epoch(self.dev_data_loader, epoch=epoch, prefix="DEV")
            _, test_evaluator = self.evaluate_epoch(self.test_data_loader, epoch=epoch, prefix="TEST")

            dev_micro_f1 = dev_evaluator.entity_metric.micro_avg_f1()
            is_curr_best = dev_micro_f1 >= self.monitor_best
            if is_curr_best:
                print("Found new best model with dev entity span-level Micro F1: {0:.3f}. "
                      "(prev. best dev Micro F1: {1:.3f})"
                      .format(dev_micro_f1, self.monitor_best))
                self.monitor_best = dev_micro_f1
            self.save_checkpoint(epoch, save_best=is_curr_best)

    def print_outputs(self, corpus, gold, predicted, mapping, outfile, mask=None):
        print("Outputs published in file: {0}".format(outfile))

        output_corpus = []
        for sent in corpus:
            output_corpus.append([EvalOutputToken(word) for word in sent])

        corpus_len = len(gold)
        for sent_index in range(corpus_len):
            spans = gold[sent_index]
            for span in spans:
                for word_index in range(span.start, span.end + 1):
                    output_corpus[sent_index][word_index].gold_tags.append(span.tag)
        for sent_index in range(corpus_len):
            spans = predicted[sent_index]
            for span in spans:
                for word_index in range(span.start, span.end + 1):
                    output_corpus[sent_index][word_index].predicted_tags.append(span.tag)
        with open(outfile, "w") as f:
            f.write("Token\tGold\tPredicted\n")
            for output_sent in output_corpus:
                for output_word in output_sent:
                    # TODO: Check how to handle PAD tags (empty text of token?)
                    if len(output_word.text) == 0:  # <PAD> token
                        continue
                    gold_tags = ",".join(sorted(output_word.gold_tags))
                    predicted_tags = ",".join(sorted(output_word.predicted_tags))
                    if len(gold_tags) == 0:
                        gold_tags = "O"
                    if len(predicted_tags) == 0:
                        predicted_tags = "O"
                    f.write("{0}\t{1}\t{2}\n".format(output_word.text, gold_tags, predicted_tags))
                f.write("\n")

    def query(self, sentence_text):
        self.model.eval()

        sentence_tokens = SpanExecutor.get_query_tokens(sentence_text)
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


class EvalOutputToken:
    def __init__(self, text):
        self.text = text
        self.gold_tags = []
        self.predicted_tags = []


def main(args):
    config = parse_config(args.config)
    set_all_seeds(config.seed)
    executor = SpanExecutor(config)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Span-CNN-LSTM Model for Sequence Labeling")
    ap.add_argument("--config", default="../configs/span-config.json", help="config file")
    ap = ap.parse_args()
    main(ap)
