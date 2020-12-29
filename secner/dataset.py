import csv
import os

import numpy as np
from torch.utils.data import Dataset

from src.utils.dataset_utils import Token


class ProcessedSentenceAndTag:

    def __init__(self, text_sentence, word_indexed_sentence, word_level_sentence_mask, text_tag):
        self.text_sentence = text_sentence
        self.word_indexed_sentence = word_indexed_sentence
        self.word_level_sentence_mask, = word_level_sentence_mask,
        self.text_tag = text_tag


class TypeDataset(Dataset):

    def __init__(self, config, corpus_type):
        super(TypeDataset, self).__init__()
        self.config = config
        self.corpus_path = self.set_corpus_path(corpus_type)
        
        self.config.data.tags_path = os.path.join(self.config.data.data_dir, self.config.data.tags_path)
        
        self.out_tags = []
        self.parse_tags()

        self.text_sentences = []
        self.word_indexed_sentences = []
        self.word_level_masks = []
        self.text_tags = []
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.use_word)
        self.parse_dataset()

    def set_corpus_path(self, corpus_type):
    	if corpus_type == "train":
            return self.config.data.train_path
        if corpus_type == "dev":
            return self.config.data.dev_path
        if corpus_type == "test":
            return self.config.data.test_path
        return None

    def parse_tags(self):
        self.out_tags.append(self.config.pad_tag)
        with open(self.config.data.tags_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tags.append(line)

    def parse_dataset(self):
        text_sentences = self.read_dataset(self.corpus_path)
        for text_sentence in text_sentences:
            self.add_processed_sentence_tag(self.process_sentence_and_tag(text_sentence))

    def add_processed_sentence_tag(self, out):
        self.text_sentences.append(out.text_sentence)
        self.word_indexed_sentences.append(out.word_indexed_sentence)
        self.word_level_masks.append(out.word_level_sentence_mask)
        self.text_tags.append(out.text_tag)

    def read_dataset(self, file_path):
        sentences = []
        with open(file_path, "r", encoding="utf-8") as f:
            sentence = []
            for line in f:
            	line = line.strip()
                if line:
                	s = line.split("\t")
                    if len(s) >= 3:
                        sentence.append(Token(start=0, text=s[0], pos_tag=s[1], dep_tag=s[2], tag=s[-1]))
                    else:
                        sentence.append(Token(start=0, text=s[0], tag=s[-1]))
                else:
                    sentences.append(sentence)
                    sentence = []
        return sentences

    def __len__(self):
        return len(self.text_sentences)

    def __getitem__(self, index):
        text_sentence = self.text_sentences[index]
        word_indexed_sentence = self.word_indexed_sentences[index]
        word_level_mask = self.word_level_masks[index]
        indexed_tag = self.get_indexed_tag(self.text_tags[index])

        return \
            text_sentence, \
            word_indexed_sentence, \
            word_level_mask, \
            indexed_tag

    def get_indexed_tag(self, text_tag):
        indexed_tag = []
        for curr_tag in text_tag:
            if curr_tag in self.out_tags:
                indexed_tag.append(self.out_tags.index(curr_tag))
            else:
                indexed_tag.append(self.out_tags.index(self.config.none_tag))
        return np.array(indexed_tag, dtype=np.int64)

    def process_sentence_and_tag(self, token_sentence):

        # check how to get masks also from here itself.. truncation to max_len may be handled here itself.. padding might be removed from here and taken care off by collate_fn! check!
        bert_batch_tokens = self.bert_tokenizer(batch_text, is_pretokenized=True, return_tensors="pt", padding="max_length",
                                           max_length=(CNN_LSTM_Base.EXPAND_FACTOR * seq_len))["input_ids"]

        if len(token_sentence) > self.config.max_seq_len:
            word_level_sentence_mask = [1] * self.config.max_seq_len
            token_sentence = token_sentence[:self.config.max_seq_len]
        else:
            if self.config.post_padding:
                word_level_sentence_mask = [1] * len(token_sentence) + [0] * (
                        self.config.max_seq_len - len(token_sentence))
                token_sentence = token_sentence + [
                    Token(start=0, text="", tag=self.config.pad_tag, pos_tag=self.config.pad_tag,
                          dep_tag=self.config.pad_tag)] * (self.config.max_seq_len - len(token_sentence))
            else:
                word_level_sentence_mask = [0] * (self.config.max_seq_len - len(token_sentence)) + [1] * len(
                    token_sentence)
                token_sentence = [Token(start=0, text="", tag=self.config.pad_tag, pos_tag=self.config.pad_tag,
                                        dep_tag=self.config.pad_tag)] * (
                                         self.config.max_seq_len - len(token_sentence)) + token_sentence
        word_indexed_sentence = []
        for index in range(len(token_sentence)):
            word = token_sentence[index]
            if word_level_sentence_mask[index] == 0:
                word_index = [self.word_vocab_index[self.config.pad_tag], self.pos_tag_vocab.index(word.pos_tag),
                              self.dep_tag_vocab.index(word.dep_tag)]
                word_indexed_sentence.append(word_index)  # pad tag
            else:
                lw = word.text.lower()
                if lw not in self.word_vocab_index:
                    lw = self.config.unk_tag  # unknown tag
                word_index = [self.word_vocab_index[lw], self.pos_tag_vocab.index(word.pos_tag),
                              self.dep_tag_vocab.index(word.dep_tag)]
                word_indexed_sentence.append(word_index)

        word_indexed_sentence = np.array(word_indexed_sentence, dtype=np.int64)
        word_level_sentence_mask = np.array(word_level_sentence_mask)

        text_sentence = [token.text for token in token_sentence]
        text_tag = [token.tag for token in token_sentence]

        return ProcessedSentenceAndTag(text_sentence, word_indexed_sentence, word_level_sentence_mask, text_tag)
