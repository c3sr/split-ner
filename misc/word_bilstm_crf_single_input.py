import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class WordBiLSTMCRF(nn.Module):

    def __init__(self, vocab_size, tag_to_index, emb_dim, hidden_dim):
        super(WordBiLSTMCRF, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_index = tag_to_index
        self.tagset_size = len(tag_to_index)

        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden_to_tag = nn.Linear(in_features=self.hidden_dim, out_features=self.tagset_size)
        self.transitions = nn.Parameter(data=torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[self.tag_to_index[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_index[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2)

    def _get_lstm_features(self, sentence):
        embed = self.emb(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embed, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden_to_tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        start_tag_tensor = torch.tensor([self.tag_to_index[START_TAG]], dtype=torch.long)
        tags = torch.cat([start_tag_tensor, tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score += self.transitions[self.tag_to_index[STOP_TAG], tags[-1]]
        return score

    def _veterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full(size=(1, self.tagset_size), fill_value=-10000.)
        init_vvars[0][self.tag_to_index[START_TAG]] = 0.

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            veterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                veterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(veterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_index[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_index[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def _forward_algo(self, feats):
        init_alphas = torch.full(size=(1, self.tagset_size), fill_value=-10000.)
        init_alphas[0][self.tag_to_index[START_TAG]] = 0.

        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + emit_score + trans_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_index[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_algo(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._veterbi_decode(lstm_feats)
        return score, tag_seq


class WordDataset(Dataset):

    def __init__(self, datapath, vocabpath, embpath=None, emb_dim=50, out_tags=None, none_tag="O", eos_tag="<EOS>",
                 unknown_tag="<UNK>", pad_tag="<PAD>", max_seq_len=20):
        super(WordDataset, self).__init__()
        self.datapath = datapath
        self.vocabpath = vocabpath
        self.embpath = embpath
        self.out_tags = out_tags
        self.none_tag = none_tag
        self.eos_tag = eos_tag
        self.unknown_tag = unknown_tag
        self.pad_tag = pad_tag
        self.max_seq_len = max_seq_len
        self.emb = []
        self.emb_dim = emb_dim  # gets overwritten if embeddings file is provided with different embedding dimensions
        self.sentences = []
        self.tags = []
        self.vocab = []
        self.vocabdict = dict()
        self.parse_vocab()
        self.parse_embfile()
        self.parse_dataset()

    def __len__(self):
        return len(self.sentences)

    def parse_vocab(self):
        with open(self.vocabpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.vocab.append(line)
        if self.unknown_tag in self.vocab:
            self.vocab.remove(self.unknown_tag)
        if self.pad_tag in self.vocab:
            self.vocab.remove(self.pad_tag)
        self.vocab += [self.unknown_tag, self.pad_tag]

        for i, word in enumerate(self.vocab):
            self.vocabdict[word] = i

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

        for key in embdict:
            self.emb_dim = embdict[key].shape[0]
            break

        emb = []
        for word in self.vocab:
            if word == self.pad_tag or word == self.unknown_tag:
                # embeddings for special tags initialized to zero-vector, if pre-trained word emb. are provided
                emb.append(np.zeros(shape=self.emb_dim, dtype=np.float32))
            else:
                # TODO: Explicitly throw exception if word of vocab not  found in provided embeddings file
                emb.append(embdict[word])
        self.emb = torch.as_tensor(np.array(emb))

    def parse_dataset(self):
        unique_out_tags = set()
        sentences = []
        tags = []
        with open(self.datapath, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            sentence = []
            tag = []
            for row in reader:
                if row[1] == self.eos_tag:
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence = []
                    tag = []
                else:
                    sentence.append(row[0].lower())
                    tag.append(row[1])
                    unique_out_tags.add(row[1])

        unique_out_tags = list(unique_out_tags)
        if self.none_tag in unique_out_tags:
            unique_out_tags.remove(self.none_tag)
        if not self.out_tags:
            self.out_tags = [self.none_tag] + unique_out_tags

        for index in range(len(sentences)):
            if len(sentences[index]) > self.max_seq_len:
                sentences[index] = sentences[index][:self.max_seq_len]
                tags[index] = tags[index][:self.max_seq_len]
            else:
                sentences[index] += [self.pad_tag] * (self.max_seq_len - len(sentences[index]))
                tags[index] += [self.none_tag] * (self.max_seq_len - len(tags[index]))

            processed_sentence = []
            for word in sentences[index]:
                if word in self.vocabdict:
                    processed_sentence.append(self.vocabdict[word])
                else:
                    processed_sentence.append(self.vocabdict[self.unknown_tag])
            self.sentences.append(np.array(processed_sentence))
            processed_tag = [self.out_tags.index(t) for t in tags[index]]
            self.tags.append(np.array(processed_tag))

    def __getitem__(self, index):
        # sentence = []
        # for word in self.sentences[index]:
        #     if word in self.emb:
        #         sentence.append(self.emb[word])
        #     else:
        #         sentence.append(np.zeros(shape=self.emb_dim, dtype=np.float32))
        # sentence = np.array(sentence)
        # tag = np.array([self.out_tags.index(t) for t in self.tags[index]])
        # return sentence, tag
        return self.sentences[index], self.tags[index]


if __name__ == "__main__":
    dataset = WordDataset("../../data/GENIA_term_3.02/processed.tsv", "../../data/GENIA_term_3.02/glove_vocab.txt",
                          "../../../../Embeddings/glove.6B.50d.txt")
    data_loader = DataLoader(dataset=dataset, batch_size=100, shuffle=False)
    dataset.out_tags.append(START_TAG)
    dataset.out_tags.append(STOP_TAG)
    tag_to_index = {k: i for i, k in enumerate(dataset.out_tags)}
    model = WordBiLSTMCRF(vocab_size=len(dataset.emb), tag_to_index=tag_to_index, emb_dim=dataset.emb_dim,
                          hidden_dim=50)
    params = filter(lambda p: p.requires_grad, model.parameters())
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(params=params, lr=0.001)

    for epoch in range(100):
        model.train()
        for batch in data_loader:
            feature, label = batch
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(feature, label)
            print("loss:", loss)
            loss.backward()
            optimizer.step()
        print("epoch done")
        # model.eval()
        # for batch in data_loader:
        #     feature, label = batch
        #     with torch.no_grad():
        #         prediction = model(feature)
        #     loss = criterion(prediction, label)
