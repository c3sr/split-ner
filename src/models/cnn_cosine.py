import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.cnn_lstm import CNN_LSTM
from src.models.base import BaseExecutor
from src.reader.type_masked_dataset import TypeMaskedDataset
from src.utils.evaluator import Evaluator
from src.utils.general import set_all_seeds


class ModelLoss(nn.Module):

    def __init__(self, tag_emb, loss_type, device):
        super(ModelLoss, self).__init__()
        self.loss_type = loss_type

        if self.loss_type == "cosine":
            self.tag_emb = torch.as_tensor(tag_emb, device=device)
            self.cos = nn.CosineSimilarity(dim=3)
            self.loss = nn.MSELoss(reduction="sum")
        else:
            self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inp, mask, target_indices):
        if self.loss_type == "cosine":
            inp = inp.transpose(1, 2)
            batch_size, seq_len, emb_dim = inp.shape
            num_tags = self.tag_emb.shape[0]

            flattened_indices = target_indices.flatten()
            target = torch.index_select(self.tag_emb, dim=0, index=flattened_indices)
            target = target.reshape(batch_size, seq_len, emb_dim)

            inp = inp.unsqueeze(2).expand(batch_size, seq_len, num_tags, emb_dim)
            target = target.unsqueeze(2).expand(batch_size, seq_len, num_tags, emb_dim)

            tag_emb = self.tag_emb.unsqueeze(0).expand((seq_len, num_tags, emb_dim))
            tag_emb = tag_emb.unsqueeze(0).expand(batch_size, seq_len, num_tags, emb_dim)

            c_inp = (1.0 - self.cos(inp, tag_emb)).transpose(1, 2)
            c_target = (1.0 - self.cos(target, tag_emb)).transpose(1, 2)

            c_inp = c_inp * mask.unsqueeze(1).expand_as(c_inp)
            c_target = c_target * mask.unsqueeze(1).expand_as(c_target)

            loss = self.loss(c_inp, c_target)

        else:
            loss = torch.sum(self.loss(inp, target_indices) * mask)

        return loss


class TypeCNN_LSTMCustomLossExecutor(BaseExecutor):

    def __init__(self, args):
        super(TypeCNN_LSTMCustomLossExecutor, self).__init__(args)

        self.args.inp_tag_vocab_path = os.path.join(self.args.data_dir, self.args.inp_tag_vocab_path)
        self.args.mask_tag_vocab_path = os.path.join(self.args.data_dir, self.args.mask_tag_vocab_path)
        self.args.tag_emb_path = os.path.join(self.args.data_dir, self.args.tag_emb_path)

        train_char_emb = self.args.use_char != "none" or self.args.use_pattern != "none"
        post_padding = not self.args.use_pre_padding
        include_word_lengths = not self.args.ignore_word_lengths
        retain_digits = not self.args.escape_digits
        use_lstm = not self.args.no_lstm

        self.train_dataset = TypeMaskedDataset(corpus_path=self.args.train_path, out_tag_vocab_path=self.args.tags_path,
                                               word_vocab_path=self.args.word_vocab_path,
                                               out_tag_names_path=self.args.out_tag_names_path,
                                               pos_tag_vocab_path=self.args.pos_tag_vocab_path,
                                               dep_tag_vocab_path=self.args.dep_tag_vocab_path,
                                               word_emb_path=self.args.emb_path,
                                               tag_emb_path=self.args.tag_emb_path,
                                               mask_tag_vocab_path=self.args.mask_tag_vocab_path,
                                               unk_tag=self.unk_tag, pad_tag=self.pad_tag, none_tag=self.none_tag,
                                               use_char=self.args.use_char, use_word=self.args.use_word,
                                               use_pattern=self.args.use_pattern, word_emb_dim=self.args.word_emb_dim,
                                               max_word_len=self.args.max_word_len, max_seq_len=self.args.max_seq_len,
                                               post_padding=post_padding, retain_digits=retain_digits,
                                               include_word_lengths=include_word_lengths,
                                               use_tag_info=self.args.use_tag_info,
                                               inp_tag_vocab_path=self.args.inp_tag_vocab_path,
                                               window_size=self.args.window_size)

        # not parsing the embedding file again when processing the dev/test sets

        self.dev_dataset = TypeMaskedDataset(corpus_path=self.args.dev_path, out_tag_vocab_path=self.args.tags_path,
                                             out_tag_names_path=self.args.out_tag_names_path,
                                             word_vocab_path=self.args.word_vocab_path, word_emb_path=None,
                                             unk_tag=self.unk_tag, mask_tag_vocab_path=self.args.mask_tag_vocab_path,
                                             pos_tag_vocab_path=self.args.pos_tag_vocab_path,
                                             dep_tag_vocab_path=self.args.dep_tag_vocab_path,
                                             tag_emb_path=self.args.tag_emb_path,
                                             pad_tag=self.pad_tag, none_tag=self.none_tag, use_char=self.args.use_char,
                                             use_pattern=self.args.use_pattern, use_word=self.args.use_word,
                                             word_emb_dim=self.args.word_emb_dim, max_word_len=self.args.max_word_len,
                                             max_seq_len=self.args.max_seq_len, post_padding=post_padding,
                                             retain_digits=retain_digits,
                                             include_word_lengths=include_word_lengths,
                                             use_tag_info=self.args.use_tag_info,
                                             inp_tag_vocab_path=self.args.inp_tag_vocab_path,
                                             window_size=self.args.window_size)

        self.test_dataset = TypeMaskedDataset(corpus_path=self.args.test_path, out_tag_vocab_path=self.args.tags_path,
                                              out_tag_names_path=self.args.out_tag_names_path,
                                              word_vocab_path=self.args.word_vocab_path,
                                              pos_tag_vocab_path=self.args.pos_tag_vocab_path,
                                              dep_tag_vocab_path=self.args.dep_tag_vocab_path, word_emb_path=None,
                                              unk_tag=self.unk_tag, mask_tag_vocab_path=self.args.mask_tag_vocab_path,
                                              tag_emb_path=self.args.tag_emb_path,
                                              pad_tag=self.pad_tag, none_tag=self.none_tag, use_char=self.args.use_char,
                                              use_pattern=self.args.use_pattern, use_word=self.args.use_word,
                                              word_emb_dim=self.args.word_emb_dim, max_word_len=self.args.max_word_len,
                                              max_seq_len=self.args.max_seq_len, post_padding=post_padding,
                                              retain_digits=retain_digits,
                                              include_word_lengths=include_word_lengths,
                                              use_tag_info=self.args.use_tag_info,
                                              inp_tag_vocab_path=self.args.inp_tag_vocab_path,
                                              window_size=self.args.window_size)

        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size,
                                            shuffle=self.shuffle_train_data)
        self.dev_data_loader = DataLoader(dataset=self.dev_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=args.batch_size, shuffle=False)

        pre_trained_emb = None
        if self.args.use_word == "glove":
            pre_trained_emb = torch.as_tensor(self.train_dataset.word_emb, device=self.device)

        out_dim = len(self.train_dataset.out_tags)
        if self.args.loss_type == "cosine":
            out_dim = self.train_dataset.out_tag_emb.shape[1]

        tag_emb = None
        if self.args.use_class_guidance or self.args.use_tag_cosine_sim:
            tag_emb = self.prep_tag_emb_tensor()

        self.model = CNN_LSTM(inp_dim=self.train_dataset.inp_dim, conv1_dim=self.args.conv1_dim,
                              out_dim=out_dim, hidden_dim=self.args.hidden_dim,
                              kernel_size=args.kernel_size, word_len=self.train_dataset.max_word_len,
                              word_vocab_size=len(self.train_dataset.word_vocab),
                              pos_tag_vocab_size=len(self.train_dataset.pos_tag_vocab),
                              dep_tag_vocab_size=len(self.train_dataset.dep_tag_vocab), use_lstm=use_lstm,
                              word_emb_dim=self.train_dataset.word_emb_dim,
                              tag_emb_dim=self.train_dataset.tag_emb_dim, pos_tag_emb_dim=self.args.pos_tag_emb_dim,
                              dep_tag_emb_dim=self.args.dep_tag_emb_dim, pre_trained_emb=pre_trained_emb,
                              use_char=train_char_emb, use_word=self.args.use_word, use_maxpool=self.args.use_maxpool,
                              use_pos_tag=self.args.use_pos_tag, use_dep_tag=self.args.use_dep_tag,
                              use_tag_info=self.args.use_tag_info, device=self.device,
                              fine_tune_bert=self.args.fine_tune_bert, use_tfo=self.args.use_tfo,
                              use_tag_cosine_sim=self.args.use_tag_cosine_sim, dropout_ratio=self.args.dropout_ratio,
                              use_class_guidance=self.args.use_class_guidance, tag_emb=tag_emb,
                              word_emb_model_from_tf=self.args.word_emb_model_from_tf,
                              num_lstm_layers=self.args.num_lstm_layers)

        self.criterion = ModelLoss(tag_emb=self.train_dataset.out_tag_emb, loss_type=self.args.loss_type,
                                   device=self.device)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=args.lr)

    def prep_tag_emb_tensor(self):
        model_training_out_tags = self.train_dataset.out_tags
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
        train_tag_mask = []
        with tqdm(self.train_data_loader) as progress_bar:
            for text, word_feature, char_feature, type_feature, word_mask, char_mask, tag_mask, label in progress_bar:
                text = np.array(text).T.tolist()
                word_feature = word_feature.to(self.device)
                char_feature = char_feature.to(self.device)
                type_feature = type_feature.to(self.device)
                word_mask = word_mask.to(self.device)
                char_mask = char_mask.to(self.device)
                tag_mask = tag_mask.to(self.device)
                label = label.to(self.device)
                batch_size = label.shape[0]
                self.optimizer.zero_grad()
                prediction = self.model(text, word_feature, char_feature, type_feature, word_mask, char_mask)
                dup_prediction = prediction.detach().clone()
                dup_tag_mask = tag_mask.detach().clone()
                dup_prediction = self.get_model_predictions(self.train_dataset, dup_prediction, dup_tag_mask)
                train_prediction.append(dup_prediction)
                train_label.append(label.detach().clone())
                train_tag_mask.append(dup_tag_mask)
                prediction = prediction.transpose(2, 1)
                loss = self.criterion(prediction, tag_mask, label)
                progress_bar.set_postfix(Epoch=epoch, Batch_Loss="{0:.3f}".format(loss.item() / batch_size))
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

        train_prediction = torch.cat(train_prediction, dim=0).cpu().numpy()
        train_label = torch.cat(train_label, dim=0).cpu().numpy()
        train_tag_mask = torch.cat(train_tag_mask, dim=0).cpu().numpy()

        evaluator = Evaluator(gold=train_label, predicted=train_prediction, tags=self.train_dataset.out_tags,
                              mask=train_tag_mask,
                              ignore_tags=[self.none_tag, self.pad_tag], none_tag=self.none_tag, pad_tag=self.pad_tag)
        print("TRAIN: Epoch: {0} | Loss:{1:.3f} | Token-Level Micro F1: {2:.3f}".format(epoch, train_loss / len(
            self.train_data_loader.dataset), evaluator.significant_token_metric.micro_avg_f1()))

    def evaluate_epoch(self, data_loader, epoch, prefix, outfile=None):
        self.model.eval()
        total_loss = 0.0
        total_text = []
        total_prediction = []
        total_tag_mask = []
        total_label = []
        for text, word_feature, char_feature, type_feature, word_mask, char_mask, tag_mask, label in data_loader:
            text = np.array(text).T.tolist()
            word_feature = word_feature.to(self.device)
            char_feature = char_feature.to(self.device)
            type_feature = type_feature.to(self.device)
            word_mask = word_mask.to(self.device)
            char_mask = char_mask.to(self.device)
            tag_mask = tag_mask.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                prediction = self.model(text, word_feature, char_feature, type_feature, word_mask, char_mask)
                dup_prediction = prediction.clone()
                dup_tag_mask = tag_mask.clone()
                dup_prediction = self.get_model_predictions(data_loader.dataset, dup_prediction, dup_tag_mask)
                total_text.extend(text)
                total_prediction.append(dup_prediction)
                total_label.append(label.clone())
                total_tag_mask.append(dup_tag_mask)
                prediction = prediction.transpose(2, 1)
                loss = self.criterion(prediction, tag_mask, label)
                total_loss += loss.item()

        total_prediction = torch.cat(total_prediction, dim=0).cpu().numpy()
        total_tag_mask = torch.cat(total_tag_mask, dim=0).cpu().numpy()
        total_label = torch.cat(total_label, dim=0).cpu().numpy()

        if outfile:
            self.print_outputs(corpus=total_text, gold=total_label, predicted=total_prediction, mask=total_tag_mask,
                               mapping=data_loader.dataset.out_tags, outfile=outfile)
        evaluator = Evaluator(gold=total_label, predicted=total_prediction, tags=data_loader.dataset.out_tags,
                              mask=total_tag_mask,
                              ignore_tags=[self.none_tag, self.pad_tag], none_tag=self.none_tag, pad_tag=self.pad_tag)
        mean_loss = total_loss / len(data_loader.dataset)
        print("{0}: Epoch: {1} | Token-Level Micro F1: {2:.3f} | Loss: {3:.3f}".format(
            prefix, epoch, evaluator.significant_token_metric.micro_avg_f1(), mean_loss))
        if self.args.verbose:
            print("Entity-Level Metrics:")
            print(evaluator.entity_metric.report())
            print("Token-Level Metrics:")
            print(evaluator.significant_token_metric.report())

        return mean_loss, evaluator

    def get_model_predictions(self, dataset, prediction, mask):
        if self.args.loss_type == "cosine":
            batch_size, seq_len, emb_dim = prediction.shape
            tag_emb = torch.as_tensor(dataset.out_tag_emb, device=self.device)
            num_tags = tag_emb.shape[0]
            tag_emb = tag_emb.unsqueeze(0).expand((seq_len, num_tags, emb_dim))
            tag_emb = tag_emb.unsqueeze(0).expand(batch_size, seq_len, num_tags, emb_dim)
            prediction = prediction.unsqueeze(2).expand(batch_size, seq_len, num_tags, emb_dim)
            cos = nn.CosineSimilarity(dim=3)
            sim = cos(prediction, tag_emb)
            prediction = torch.argmax(sim, dim=2) * mask
            return prediction

        return torch.argmax(prediction, dim=2) * mask

    def query(self, sentence_text):
        self.model.eval()

        sentence_tokens = TypeCNN_LSTMCustomLossExecutor.get_query_tokens(sentence_text)
        text, word_feature, char_feature, type_feature, word_mask, char_mask, tag_mask, _ = self.test_dataset.get_query_given_tokens(
            sentence_tokens)
        text = [text]
        word_feature = torch.as_tensor(word_feature, device=self.device).unsqueeze(0)
        char_feature = torch.as_tensor(char_feature, device=self.device).unsqueeze(0)
        type_feature = torch.as_tensor(type_feature, device=self.device).unsqueeze(0)
        word_mask = torch.as_tensor(word_mask, device=self.device).unsqueeze(0)
        char_mask = torch.as_tensor(char_mask, device=self.device).unsqueeze(0)
        tag_mask = torch.as_tensor(tag_mask, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(text, word_feature, char_feature, type_feature, word_mask, char_mask, tag_mask)
        prediction = np.argmax(prediction, axis=2).squeeze(0)
        for i in range(prediction.shape[0]):
            print("{0}\t{1}".format(text[0][i], self.test_dataset.out_tags[prediction[i]]))


def main(args):
    set_all_seeds(args.seed)
    executor = TypeCNN_LSTMCustomLossExecutor(args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Type-CNN Model for Sequence Labeling (POC)")
    ap.add_argument("--name", type=str, default="cnn-cosine",
                    help="model name (Default: 'cnn-cosine')")
    ap.add_argument("--checkpoint_dir", type=str, default="../../checkpoints",
                    help="checkpoints directory (Default: '../../checkpoints')")
    ap.add_argument("--eval", type=str, default="none",
                    help="only evaluate existing checkpoint model (none/best/<checkpoint-id>) (Default: 'none')")
    ap.add_argument("--query", action="store_true",
                    help="query mode, can be used with eval to work with best model (Default: False)")

    ap.add_argument("--data_dir", type=str, default="../../data/GENIA_term_3.02",
                    help="path to input dataset directory (Default: '../../data/GENIA_term_3.02')")
    ap.add_argument("--out_dir", type=str, default="../../data/GENIA_term_3.02/out",
                    help="path to output directory (Default: '../../data/GENIA_term_3.02/out')")
    ap.add_argument("--train_path", type=str, default="train.tsv",
                    help="path to train dataset (train.tsv|std_train.tsv|jnlpba_train.tsv) (Default: 'train.tsv')")
    ap.add_argument("--dev_path", type=str, default="dev.tsv",
                    help="path to dev dataset (dev.tsv|std_dev.tsv|jnlpba_dev.tsv) (Default: 'dev.tsv')")
    ap.add_argument("--test_path", type=str, default="test.tsv",
                    help="path to test dataset (test.tsv|std_test.tsv|jnlpba_test.tsv) (Default: 'test.tsv')")
    ap.add_argument("--word_vocab_path", type=str, default="glove_vocab.txt",
                    help="path to word vocab (Default: 'glove_vocab.txt')")
    ap.add_argument("--tags_path", type=str, default="tag_vocab_two_step.txt",
                    help="path to output tags vocab. Use 'tag_vocab.txt' for full tags vocab. "
                         "Use 'std_tag_vocab.txt' for standard 5 tags vocab. "
                         "Use 'jnlpba_tag_vocab.tsv' for exact (5-tag) settings used by MTL-BioInformatics-2016 "
                         "(ref: https://github.com/cambridgeltl/MTL-Bioinformatics-2016)"
                         "Use 'out_freq_tag_vocab.txt' for reduced tags, when considering input tags information. "
                         "Use 'out_freq_tag_vocab_two_step.txt' for reduced tags, when considering two-step labelling "
                         "and input tag information. "
                         "tag_vocab_two_step.txt' for reduced tags, when considering two-step labelling. "
                         "(Default: 'tag_vocab_two_step.txt')")
    ap.add_argument("--out_tag_names_path", type=str, default="tag_names_two_step.txt",
                    help="path to output tags vocab. Use 'tag_names.txt' for full tags vocab. "
                         "Use 'std_tag_names.txt' for standard 5 tags vocab. "
                         "Use 'jnlpba_tag_names.tsv' for exact (5-tag) settings used by MTL-BioInformatics-2016 "
                         "(ref: https://github.com/cambridgeltl/MTL-Bioinformatics-2016)"
                         "Use 'out_freq_tag_names.txt' for reduced tags, when considering input tags information. "
                         "Use 'out_freq_tag_names_two_step.txt' for reduced tags, when considering two-step labelling "
                         "and input tag information. "
                         "tag_names_two_step.txt' for reduced tags, when considering two-step labelling. "
                         "(Default: 'tag_names_two_step.txt')")
    ap.add_argument("--inp_tag_vocab_path", type=str, default="empty_inp_tag_vocab.txt",
                    help="path to input tags vocab. Use 'empty_inp_tag_vocab.txt' if don't want to use tag info. "
                         "Use 'inp_freq_tag_vocab.txt' for specifying default input tag info."
                         "(Default: 'empty_inp_tag_vocab.txt')")
    ap.add_argument("--mask_tag_vocab_path", type=str, default="mask_freq_tag_vocab.txt",
                    help="path to masked tags vocab. Use 'empty_mask_tag_vocab.txt' if don't want to mask out any tag. "
                         "Use 'mask_freq_tag_vocab.txt' for masking tags which are very less frequent in the dataset, "
                         "including some special tags that need to be separately handled."
                         "(Default: 'mask_freq_tag_vocab.txt')")
    ap.add_argument("--pos_tag_vocab_path", type=str, default="pos_tag_vocab.txt",
                    help="path to POS tags vocab. (pos_tag_vocab.txt|jnlpba_pos_tag_vocab.txt) "
                         "(Default: 'pos_tag_vocab.txt')")
    ap.add_argument("--dep_tag_vocab_path", type=str, default="dep_tag_vocab.txt",
                    help="path to dependency-parse tags vocab. (dep_tag_vocab.txt|jnlpba_dep_tag_vocab.txt) "
                         "(Default: 'dep_tag_vocab.txt')")
    ap.add_argument("--emb_path", type=str, default="../../../../Embeddings/glove.6B.50d.txt",
                    help="path to pre-trained word embeddings (Default: '../../../../Embeddings/glove.6B.50d.txt')")
    ap.add_argument("--tag_emb_path", type=str, default="tag_w2v_emb.txt",
                    help="path to pre-trained tag embeddings, relative to data_dir "
                         "(jnlpba_tag_w2v_emb.txt|jnlpba_tag_use_emb.txt|jnlpba_tag_full_emb.txt) "
                         "(std_tag_w2v_emb.txt|std_tag_use_emb.txt|std_tag_full_emb.txt) "
                         "(tag_w2v_emb.txt|tag_use_emb.txt|tag_full_emb.txt) (Default: 'tag_w2v_emb.txt')")

    ap.add_argument("--num_epochs", type=int, default=500, help="# epochs to train (Default: 500)")
    ap.add_argument("--batch_size", type=int, default=128, help="batch size (Default: 128)")
    ap.add_argument("--word_emb_dim", type=int, default=50, help="word embedding dimension (Default: 50)")
    ap.add_argument("--pos_tag_emb_dim", type=int, default=15, help="POS tag embedding dimension (Default: 15)")
    ap.add_argument("--dep_tag_emb_dim", type=int, default=15, help="dep-parse tag embedding dimension (Default: 15)")
    ap.add_argument("--max_word_len", type=int, default=30, help="max. #chars in word (Default: 30)")
    ap.add_argument("--max_seq_len", type=int, default=60, help="max. #words in sentence (Default: 60)")
    ap.add_argument("--conv1_dim", type=int, default=128, help="conv1 layer output channels (Default: 128)")
    ap.add_argument("--hidden_dim", type=int, default=256, help="hidden state dim for LSTM, if used (Default: 256)")
    ap.add_argument("--use_maxpool", action="store_true", help="max pool over CNN output to get char embeddings, else "
                                                               "does concatenation (Default: False)")
    ap.add_argument("--use_pos_tag", action="store_true", help="embed POS tag information (Default: False)")
    ap.add_argument("--use_dep_tag", action="store_true", help="embed dep-parse tag information (Default: False)")
    ap.add_argument("--kernel_size", type=int, default=5, help="kernel size for CNN (Default: 5)")
    ap.add_argument("--num_lstm_layers", type=int, default=1, help="no. of LSTM layers (Default: 1)")
    ap.add_argument("--use_char", type=str, default="lower",
                    help="char embedding type (none/lower/all) (Default: 'lower')")
    ap.add_argument("--use_pattern", type=str, default="condensed",
                    help="pattern embedding type (none/one-to-one/condensed) (Default: 'condensed')")
    ap.add_argument("--escape_digits", action="store_true",
                    help="replace digits(0-9) with 'd' tag in pattern capturing (Default: False)")
    ap.add_argument("--ignore_word_lengths", action="store_true",
                    help="ignore word lengths in pattern capturing (Default: False)")
    ap.add_argument("--no_lstm", action="store_true",
                    help="don't use LSTM to capture neighbor context. Directly CRF over individual token level CNN "
                         "(Default: False)")
    ap.add_argument("--use_tag_cosine_sim", action="store_true",
                    help="compute cosine sim with tag embeddings as additional layer in model (Default: False)")
    ap.add_argument("--use_tag_info", type=str, default="none",
                    help="type information (none/self/window/pretrained) (Default: 'none')")
    ap.add_argument("--use_tfo", type=str, default="none",
                    help="use transformer (may not use LSTM then). 'simple' creates a basic tfo. "
                         "'xl' uses TransformerXL model layer. (none|simple|xl) (Default: 'none')")
    ap.add_argument("--window_size", type=int, default=5,
                    help="size of context window for type info on either side of current token (Default: 5)")
    ap.add_argument("--use_word", type=str, default="allenai/scibert_scivocab_uncased",
                    help="use word(token) embeddings "
                         "(none|rand|glove|allenai/scibert_scivocab_uncased|bert-base-uncased"
                         "|../../../resources/biobert_v1.1_pubmed) "
                         "(Default: allenai/scibert_scivocab_uncased)")
    ap.add_argument("--loss_type", type=str, default="cosine",
                    help="loss function (cross_entropy/cosine) (Default: cosine)")
    ap.add_argument("--use_pre_padding", action="store_true", help="pre-padding for char/word (Default: False)")
    ap.add_argument("--word_emb_model_from_tf", action="store_true",
                    help="word embedding generator model is a pretrained tensorflow model. Use 'True' for models like, "
                         "'../../../resources/biobert_v1.1_pubmed' (Default: False)")
    ap.add_argument("--use_class_guidance", action="store_true",
                    help="take guidance through pre-trained class embeddings (Default: False)")
    ap.add_argument("--fine_tune_bert", action="store_true", help="fine-tune bert embeddings (Default: False)")
    ap.add_argument("--lr", type=float, default=0.001, help="learning rate (Default: 0.001)")
    ap.add_argument("--dropout_ratio", type=float, default=0.5, help="dropout ratio (Default: 0.5)")
    ap.add_argument("--seed", type=int, default=42, help="manual seed for reproducibility (Default: 42)")
    ap.add_argument("--use_cpu", action="store_true", help="force CPU usage (Default: False)")
    ap.add_argument("--no_eval_print", action="store_true",
                    help="don't output verbose evaluation matrices (Default: False)")
    ap = ap.parse_args()
    main(ap)
