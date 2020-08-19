import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"


def log_sum_exp(vec):
    max_score, _ = torch.max(vec, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(vec)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), -1))


class CRF(nn.Module):

    def __init__(self, tag_to_index):
        super(CRF, self).__init__()
        self.tag_to_index = tag_to_index
        self.tagset_size = len(self.tag_to_index)
        self.transitions = nn.Parameter(data=torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[self.tag_to_index[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_index[STOP_TAG]] = -10000
        self.transitions.data[:, self.tag_to_index[PAD_TAG]] = -10000
        self.transitions.data[self.tag_to_index[PAD_TAG], :] = -10000

        self.transitions.data[self.tag_to_index[PAD_TAG], self.tag_to_index[STOP_TAG]] = 0
        self.transitions.data[self.tag_to_index[PAD_TAG], self.tag_to_index[PAD_TAG]] = 0

    def score_sentence(self, feats, tags, masks):
        """
        :param feats: [B, S, T]
        :param tags: [B, S]
        :param masks: [B, S]
        :return:
        """
        batch_size, seq_len = masks.shape
        score = torch.zeros(batch_size, dtype=torch.float32)  # [B]
        start_tag_tensor = torch.full(size=(batch_size, 1), fill_value=self.tag_to_index[START_TAG],
                                      dtype=torch.long)  # [B, 1]
        tags = torch.cat([start_tag_tensor, tags], 1)  # [B, S+1]

        for i in range(seq_len):
            curr_mask = masks[:, i]
            curr_emission = torch.zeros(batch_size, dtype=torch.float32)  # [B]
            curr_transition = torch.zeros(batch_size, dtype=torch.float32)  # [B]
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
        backpointers = torch.LongTensor()
        init_vvars = torch.full(size=(batch_size, self.tagset_size), fill_value=-10000.)
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

        return best_score, best_path

    def forward_algo(self, feats, masks):
        """
        :param feats: [B, S, T]
        :param masks: [B, S]
        :return:
        """
        batch_size, seq_len = masks.shape
        score = torch.full(size=(batch_size, self.tagset_size), fill_value=-10000.)
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

    def __init__(self, vocab_size, tag_to_index, emb_dim, hidden_dim):
        super(WordBiLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_index = tag_to_index
        self.tagset_size = len(tag_to_index)
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.hidden_to_tag = nn.Linear(in_features=self.hidden_dim, out_features=self.tagset_size)

    def init_hidden(self, batch_size):
        return torch.randn(2, batch_size, self.hidden_dim // 2), torch.randn(2, batch_size, self.hidden_dim // 2)

    def forward(self, sentences, masks):
        # TODO: check LSTM state initialization. Should it not be done only once in __init__?
        batch_size, seq_len = sentences.shape
        hidden = self.init_hidden(batch_size)
        embed = self.emb(sentences)
        packed_lstm_inp = nn.utils.rnn.pack_padded_sequence(input=embed, lengths=masks.sum(1).int(), batch_first=True,
                                                            enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_lstm_inp, hidden)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(sequence=packed_lstm_out, batch_first=True)
        lstm_feats = self.hidden_to_tag(lstm_out)
        lstm_feats *= masks.unsqueeze(2)
        return lstm_feats


class WordBiLSTMCRF(nn.Module):

    def __init__(self, vocab_size, tag_to_index, emb_dim, hidden_dim):
        super(WordBiLSTMCRF, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_index = tag_to_index
        self.tagset_size = len(tag_to_index)

        self.lstm = WordBiLSTM(vocab_size=self.vocab_size, tag_to_index=self.tag_to_index, emb_dim=self.emb_dim,
                               hidden_dim=self.hidden_dim)
        self.crf = CRF(tag_to_index=self.tag_to_index)

    def neg_log_likelihood(self, sentences, tags, masks):
        # masks = sentences.where(sentences != self.tag_to_index[PAD_TAG], torch.tensor([1.]), torch.tensor([0.]))
        feats = self.lstm(sentences, masks)
        forward_score = self.crf.forward_algo(feats, masks)
        gold_score = self.crf.score_sentence(feats, tags, masks)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentences, masks):
        # masks = sentences.where(sentences != self.tag_to_index[PAD_TAG], torch.tensor([1.]), torch.tensor([0.]))
        lstm_feats = self.lstm(sentences, masks)
        score, tag_seq = self.crf.veterbi_decode(lstm_feats, masks)
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
        self.masks = []
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
        if not self.out_tags:
            self.out_tags = unique_out_tags
        if self.none_tag in self.out_tags:
            self.out_tags.remove(self.none_tag)
        if self.pad_tag in self.out_tags:
            self.out_tags.remove(self.pad_tag)
        self.out_tags = [self.pad_tag, self.none_tag] + self.out_tags

        for index in range(len(sentences)):
            if len(sentences[index]) > self.max_seq_len:
                sentences[index] = sentences[index][:self.max_seq_len]
                tags[index] = tags[index][:self.max_seq_len]
                mask = [1] * self.max_seq_len
            else:
                mask = [1] * len(sentences[index]) + [0] * (self.max_seq_len - len(sentences[index]))
                sentences[index] += [self.pad_tag] * (self.max_seq_len - len(sentences[index]))
                tags[index] += [self.pad_tag] * (self.max_seq_len - len(tags[index]))

            self.masks.append(np.array(mask))
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
        return self.sentences[index], self.tags[index], self.masks[index]


if __name__ == "__main__":
    dataset = WordDataset("../../../GENIA_term_3.02/processed.tsv", "../../../GENIA_term_3.02/glove_vocab.txt",
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
            feature, label, mask = batch
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(feature, label, mask)
            loss.backward()
            optimizer.step()
            print("loss:", loss)

        print("epoch done")

        model.eval()
        for batch in data_loader:
            feature, label, mask = batch

            with torch.no_grad():
                score, prediction = model(feature, mask)
                print("prediction:", prediction)
