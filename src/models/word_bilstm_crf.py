import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.base import BaseExecutor
from src.utils.evaluator import Evaluator
from src.utils.general import set_all_seeds, log_sum_exp

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"


class CRF(nn.Module):

    def __init__(self, tag_to_index, device):
        super(CRF, self).__init__()
        self.tag_to_index = tag_to_index
        self.device = device
        self.tagset_size = len(self.tag_to_index)
        self.transitions = nn.Parameter(data=torch.randn(self.tagset_size, self.tagset_size, device=self.device))

        self.transitions.data[self.tag_to_index[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_index[STOP_TAG]] = -10000
        self.transitions.data[:, self.tag_to_index[PAD_TAG]] = -10000
        self.transitions.data[self.tag_to_index[PAD_TAG], :] = -10000

        self.transitions.data[self.tag_to_index[PAD_TAG], self.tag_to_index[STOP_TAG]] = 0
        self.transitions.data[self.tag_to_index[PAD_TAG], self.tag_to_index[PAD_TAG]] = 0

    def score_sentence(self, feats, masks, tags):
        """
        :param feats: [B, S, T]
        :param tags: [B, S]
        :param masks: [B, S]
        :return:
        """
        batch_size, seq_len = masks.shape
        score = torch.zeros(batch_size, dtype=torch.float32, device=self.device)  # [B]
        start_tag_tensor = torch.full(size=(batch_size, 1), fill_value=self.tag_to_index[START_TAG], dtype=torch.long,
                                      device=self.device)  # [B, 1]
        tags = torch.cat([start_tag_tensor, tags], 1)  # [B, S+1]

        for i in range(seq_len):
            curr_mask = masks[:, i]
            curr_emission = torch.zeros(batch_size, dtype=torch.float32, device=self.device)  # [B]
            curr_transition = torch.zeros(batch_size, dtype=torch.float32, device=self.device)  # [B]
            for j in range(batch_size):
                curr_emission[j] = feats[j, i, tags[j, i + 1]].unsqueeze(0)
                curr_transition[j] = self.transitions[tags[j, i + 1], tags[j, i]].unsqueeze(0)
            score += (curr_emission + curr_transition) * curr_mask
        last_tags = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        score += self.transitions[self.tag_to_index[STOP_TAG], last_tags]
        return score

    def veterbi_decode(self, feats, masks):
        """
        :param feats: [B, S, T]
        :param masks: [B, S]
        :return:
        """
        batch_size, seq_len = masks.shape
        backpointers = torch.LongTensor().to(self.device)
        init_vvars = torch.full(size=(batch_size, self.tagset_size), fill_value=-10000., device=self.device)
        init_vvars[:, self.tag_to_index[START_TAG]] = 0.

        for s in range(seq_len):
            curr_masks = masks[:, s].unsqueeze(1)
            curr_vvars = init_vvars.unsqueeze(1) + self.transitions  # [B, 1, T] -> [B, T, T]
            curr_vvars, curr_bptrs = curr_vvars.max(2)
            curr_vvars += feats[:, s]
            backpointers = torch.cat((backpointers, curr_bptrs.unsqueeze(1)), 1)
            init_vvars = curr_vvars * curr_masks + init_vvars * (1 - curr_masks)
        init_vvars += self.transitions[self.tag_to_index[STOP_TAG]]
        best_score, best_tag_id = init_vvars.max(1)

        backpointers = backpointers.tolist()
        best_path = [[p] for p in best_tag_id.tolist()]

        for b in range(batch_size):
            n = masks[b].sum().long().item()
            t = best_tag_id[b]
            for curr_bptrs in reversed(backpointers[b][:n]):
                t = curr_bptrs[t]
                best_path[b].append(t)
            start = best_path[b].pop()
            assert start == self.tag_to_index[START_TAG]
            best_path[b].reverse()
            best_path[b] += [self.tag_to_index[PAD_TAG]] * (seq_len - len(best_path[b]))

        return best_score, best_path

    def forward_algo(self, feats, masks):
        """
        :param feats: [B, S, T]
        :param masks: [B, S]
        :return:
        """
        batch_size, seq_len = masks.shape
        score = torch.full(size=(batch_size, self.tagset_size), fill_value=-10000., device=self.device)
        score[:, self.tag_to_index[START_TAG]] = 0.

        for s in range(seq_len):
            curr_mask = masks[:, s].unsqueeze(-1).expand_as(score)  # [B, T]
            curr_score = score.unsqueeze(1).expand(-1, *self.transitions.size())  # [B, T, T]
            curr_emission = feats[:, s].unsqueeze(-1).expand_as(curr_score)
            curr_transition = self.transitions.unsqueeze(0).expand_as(curr_score)
            curr_score = log_sum_exp(curr_score + curr_emission + curr_transition)
            score = curr_score * curr_mask + score * (1 - curr_mask)

        score = log_sum_exp(score + self.transitions[self.tag_to_index[STOP_TAG]])
        return score


class WordBiLSTM(nn.Module):

    def __init__(self, vocab_size, tag_to_index, emb_dim, hidden_dim, device, pre_trained_emb=None):
        super(WordBiLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.device = device
        self.tag_to_index = tag_to_index
        self.tagset_size = len(tag_to_index)
        if pre_trained_emb != None:
            self.emb = nn.Embedding.from_pretrained(embeddings=pre_trained_emb, freeze=False)
        else:
            self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.hidden_to_tag = nn.Linear(in_features=self.hidden_dim, out_features=self.tagset_size)

    def init_hidden(self, batch_size):
        return torch.randn(2, batch_size, self.hidden_dim // 2, device=self.device), \
               torch.randn(2, batch_size, self.hidden_dim // 2, device=self.device)

    def forward(self, sentences, masks):
        # TODO: check LSTM state initialization. Should it not be done only once in __init__?
        batch_size, seq_len = sentences.shape
        hidden = self.init_hidden(batch_size)
        embed = self.emb(sentences)
        packed_lstm_inp = nn.utils.rnn.pack_padded_sequence(input=embed, lengths=masks.sum(1).int(), batch_first=True,
                                                            enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_lstm_inp, hidden)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(sequence=packed_lstm_out, batch_first=True, total_length=seq_len)
        lstm_feats = self.hidden_to_tag(lstm_out)
        lstm_feats *= masks.unsqueeze(2)
        return lstm_feats


class WordBiLSTMCRF(nn.Module):

    def __init__(self, vocab_size, tag_to_index, emb_dim, hidden_dim, device, pre_trained_emb=None):
        super(WordBiLSTMCRF, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.device = device
        self.tag_to_index = tag_to_index
        self.tagset_size = len(tag_to_index)

        self.lstm = WordBiLSTM(vocab_size=self.vocab_size, tag_to_index=self.tag_to_index, emb_dim=self.emb_dim,
                               hidden_dim=self.hidden_dim, device=self.device, pre_trained_emb=pre_trained_emb)
        self.crf = CRF(tag_to_index=self.tag_to_index, device=self.device)

    def neg_log_likelihood(self, sentences, masks, tags):
        feats = self.lstm(sentences, masks)
        forward_score = self.crf.forward_algo(feats, masks)
        gold_score = self.crf.score_sentence(feats, masks, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentences, masks):
        lstm_feats = self.lstm(sentences, masks)
        score, tag_seq = self.crf.veterbi_decode(lstm_feats, masks)
        return score, tag_seq


class WordDataset(Dataset):

    def __init__(self, datapath, vocabpath, tagspath, embpath=None, word_emb_dim=50, none_tag="O", unk_tag="<UNK>",
                 pad_tag="<PAD>", max_seq_len=20, post_padding=True):
        super(WordDataset, self).__init__()
        self.datapath = datapath
        self.vocabpath = vocabpath
        self.tagspath = tagspath
        self.embpath = embpath
        self.none_tag = none_tag
        self.unk_tag = unk_tag
        self.pad_tag = pad_tag
        self.max_seq_len = max_seq_len
        self.post_padding = post_padding
        self.word_emb = []
        self.word_emb_dim = word_emb_dim
        self.text_sentences = []
        self.word_indexed_sentences = []
        self.word_level_masks = []
        self.tags = []
        self.word_vocab = []
        self.word_vocab_index = dict()
        self.out_tags = []
        self.parse_tags()
        self.parse_vocab()
        self.parse_embfile()
        self.parse_dataset()

    def __len__(self):
        return len(self.word_indexed_sentences)

    def parse_tags(self):
        with open(self.tagspath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tags.append(line)
        self.out_tags.append(self.pad_tag)
        self.out_tags.append(START_TAG)
        self.out_tags.append(STOP_TAG)

    def parse_vocab(self):
        with open(self.vocabpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.word_vocab.append(line)

        self.word_vocab.append(self.unk_tag)
        self.word_vocab.append(self.pad_tag)

        for i, word in enumerate(self.word_vocab):
            self.word_vocab_index[word] = i

    def parse_embfile(self):
        if not self.embpath:
            return
        embdict = dict()
        with open(self.embpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = line.split(" ")
                    embdict[s[0]] = np.array([float(x) for x in s[1:]], dtype=np.float32)
        embdict[self.unk_tag] = np.zeros(shape=self.word_emb_dim, dtype=np.float32)
        embdict[self.pad_tag] = np.zeros(shape=self.word_emb_dim, dtype=np.float32)
        self.word_emb = np.array([embdict[word] for word in self.word_vocab])

    def parse_dataset(self):
        text_sentences = []
        text_tags = []
        with open(self.datapath, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            text_sentence = []
            text_tag = []
            for row in reader:
                if len(row) == 0:
                    text_sentences.append(text_sentence)
                    text_tags.append(text_tag)
                    text_sentence = []
                    text_tag = []
                else:
                    text_sentence.append(row[0].lower())
                    text_tag.append(row[1])

        for text_sentence, text_tag in zip(text_sentences, text_tags):
            text_sentence, word_indexed_sentence, word_level_sentence_mask, indexed_tag = self.process_sentence_and_tag(
                text_sentence, text_tag)
            self.text_sentences.append(text_sentence)
            self.word_indexed_sentences.append(word_indexed_sentence)
            self.word_level_masks.append(word_level_sentence_mask)
            self.tags.append(indexed_tag)

    def process_sentence_and_tag(self, text_sentence, text_tag):
        if len(text_sentence) > self.max_seq_len:
            word_level_sentence_mask = [1] * self.max_seq_len
            text_sentence = text_sentence[:self.max_seq_len]
            text_tag = text_tag[:self.max_seq_len]
        else:
            if self.post_padding:
                word_level_sentence_mask = [1] * len(text_sentence) + [0] * (self.max_seq_len - len(text_sentence))
                text_sentence = text_sentence + [""] * (self.max_seq_len - len(text_sentence))
                text_tag = text_tag + [self.pad_tag] * (self.max_seq_len - len(text_tag))
            else:
                word_level_sentence_mask = [0] * (self.max_seq_len - len(text_sentence)) + [1] * len(text_sentence)
                text_sentence = [""] * (self.max_seq_len - len(text_sentence)) + text_sentence
                text_tag = [self.pad_tag] * (self.max_seq_len - len(text_tag)) + text_tag
        word_indexed_sentence = []
        for word in text_sentence:
            if word in self.word_vocab_index:
                word_indexed_sentence.append(self.word_vocab_index[word])
            else:
                word_indexed_sentence.append(self.word_vocab_index[self.unk_tag])
        word_indexed_sentence = np.array(word_indexed_sentence)
        word_level_sentence_mask = np.array(word_level_sentence_mask)
        indexed_tag = np.array([self.out_tags.index(t) for t in text_tag])
        return text_sentence, word_indexed_sentence, word_level_sentence_mask, indexed_tag

    def __getitem__(self, index):
        return self.text_sentences[index], self.word_indexed_sentences[index], self.word_level_masks[index], \
               self.tags[index]

    def get_query_given_tokens(self, text_sentence):
        text_tag = []
        return self.process_sentence_and_tag(text_sentence, text_tag)


class WordBiLSTMCRFExecutor(BaseExecutor):

    def __init__(self, args):
        super(WordBiLSTMCRFExecutor, self).__init__(args)

        post_padding = not self.args.use_pre_padding

        self.train_dataset = WordDataset(datapath=self.args.train_path, vocabpath=self.args.word_vocab_path,
                                         tagspath=self.args.tags_path, embpath=self.args.emb_path,
                                         word_emb_dim=self.args.word_emb_dim, unk_tag=self.unk_tag,
                                         pad_tag=self.pad_tag, max_seq_len=self.args.max_seq_len,
                                         post_padding=post_padding)
        self.dev_dataset = WordDataset(datapath=self.args.dev_path, vocabpath=self.args.word_vocab_path,
                                       tagspath=self.args.tags_path, embpath=None, word_emb_dim=self.args.word_emb_dim,
                                       unk_tag=self.unk_tag, pad_tag=self.pad_tag, max_seq_len=self.args.max_seq_len,
                                       post_padding=post_padding)
        self.test_dataset = WordDataset(datapath=self.args.test_path, vocabpath=self.args.word_vocab_path,
                                        tagspath=self.args.tags_path, embpath=None, word_emb_dim=self.args.word_emb_dim,
                                        unk_tag=self.unk_tag, pad_tag=self.pad_tag, max_seq_len=self.args.max_seq_len,
                                        post_padding=post_padding)

        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=self.dev_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=self.args.batch_size, shuffle=True)

        pre_trained_emb = None
        if not self.args.rand_embedding:
            pre_trained_emb = torch.as_tensor(self.train_dataset.word_emb, device=self.device)

        self.tag_to_index = {k: i for i, k in enumerate(self.train_dataset.out_tags)}
        self.model = WordBiLSTMCRF(vocab_size=len(self.train_dataset.word_vocab), tag_to_index=self.tag_to_index,
                                   emb_dim=self.train_dataset.word_emb_dim, hidden_dim=self.args.hidden_dim,
                                   device=self.device, pre_trained_emb=pre_trained_emb)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=args.lr)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        with tqdm(self.train_data_loader) as progress_bar:
            for text, feature, mask, label in progress_bar:
                feature = feature.to(self.device)
                mask = mask.to(self.device)
                label = label.to(self.device)
                batch_size = label.shape[0]
                self.optimizer.zero_grad()
                loss = self.model.neg_log_likelihood(feature, mask, label)
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
        for text, feature, mask, label in data_loader:
            text = np.array(text).T.tolist()
            feature = feature.to(self.device)
            mask = mask.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                score, prediction = self.model(feature, mask)
                total_text.extend(text)
                total_prediction.append(prediction)
                total_label.append(label.clone())
                total_score += score.sum().item()
        total_prediction = np.vstack([np.array(p) for p in total_prediction])
        total_label = torch.cat(total_label, dim=0).cpu().numpy()
        if outfile:
            self.print_outputs(corpus=total_text, gold=total_label, predicted=total_prediction,
                               mapping=data_loader.dataset.out_tags, outfile=outfile)
        evaluator = Evaluator(gold=total_label, predicted=total_prediction, tags=data_loader.dataset.out_tags,
                              ignore_tags=[self.none_tag, self.pad_tag], none_tag=self.none_tag, pad_tag=self.pad_tag)
        mean_score = total_score / len(data_loader.dataset.out_tags)
        print("{0}: Epoch: {1} | Token-Level Micro F1: {2:.3f} | Score: {3:.3f}".format(
            prefix, epoch, evaluator.significant_token_metric.micro_avg_f1(), mean_score))
        print("Entity-Level Metrics:")
        print(evaluator.entity_metric.report())
        print("Token-Level (Without 'O' Tag) Metrics:")
        print(evaluator.significant_token_metric.report())

        return mean_score, evaluator

    def query(self, sentence_text):
        self.model.eval()

        sentence_tokens = WordBiLSTMCRFExecutor.get_query_tokens(sentence_text)
        text, feature, mask, _ = self.test_dataset.get_query_given_tokens(sentence_tokens)
        feature = torch.as_tensor(feature, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(mask, device=self.device).unsqueeze(0)
        with torch.no_grad():
            score, prediction = self.model(feature, mask)
        prediction = prediction[0]
        for i in range(len(prediction)):
            print("{0}\t{1}".format(text[i], self.test_dataset.out_tags[prediction[i]]))


def main(args):
    set_all_seeds(args.seed)
    executor = WordBiLSTMCRFExecutor(args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Word-BiLSTM-CRF Model for Sequence Labeling")
    ap.add_argument("--name", type=str, default="bilstm-crf", help="model name (Default: 'bilstm-crf')")
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
                    help="path to train dataset (Default: 'train.tsv')")
    ap.add_argument("--dev_path", type=str, default="dev.tsv",
                    help="path to dev dataset (Default: 'dev.tsv')")
    ap.add_argument("--test_path", type=str, default="test.tsv",
                    help="path to test dataset (Default: 'test.tsv')")
    ap.add_argument("--word_vocab_path", type=str, default="glove_vocab.txt",
                    help="path to word vocab (Default: 'glove_vocab.txt')")
    ap.add_argument("--tags_path", type=str, default="tag_vocab.txt",
                    help="path to output tags vocab (Default: 'tag_vocab.txt')")
    ap.add_argument("--emb_path", type=str, default="../../../../Embeddings/glove.6B.50d.txt",
                    help="path to pre-trained word embeddings (Default: '../../../../Embeddings/glove.6B.50d.txt')")

    ap.add_argument("--num_epochs", type=int, default=500, help="# epochs to train (Default: 500)")
    ap.add_argument("--word_emb_dim", type=int, default=50, help="word embedding dimension (Default: 50)")
    ap.add_argument("--max_seq_len", type=int, default=60, help="max. #words in sentence (Default: 60)")
    ap.add_argument("--batch_size", type=int, default=128, help="batch size (Default: 128)")
    ap.add_argument("--hidden_dim", type=int, default=512, help="LSTM hidden state dimension (Default: 512)")
    ap.add_argument("--use_pre_padding", action="store_true", help="pre-padding for char/word (Default: False)")
    ap.add_argument("--rand_embedding", action="store_true",
                    help="randomly initialize word embeddings (Default: False)")
    ap.add_argument("--lr", type=float, default=0.001, help="learning rate (Default: 0.001)")
    ap.add_argument("--seed", type=int, default=42, help="manual seed for reproducibility (Default: 42)")
    ap.add_argument("--use_cpu", action="store_true", help="force CPU usage (Default: False)")
    ap = ap.parse_args()
    main(ap)
