import csv
import os

import numpy as np
from torch.utils.data import Dataset

from src.utils.char_parsers import LowercaseCharParser, CharParser, OneToOnePatternParser, WordCondensedPatternParser
from src.utils.dataset_utils import Token
from src.utils.general import parse_emb_file


class ProcessedSentenceAndTag:

    def __init__(self, text_sentence, char_indexed_sentence, pattern_indexed_sentence, word_indexed_sentence,
                 type_indexed_sentence, word_level_sentence_mask, char_level_sentence_mask, text_tag):
        self.text_sentence = text_sentence
        self.char_indexed_sentence = char_indexed_sentence
        self.pattern_indexed_sentence, = pattern_indexed_sentence,
        self.word_indexed_sentence = word_indexed_sentence
        self.type_indexed_sentence = type_indexed_sentence
        self.word_level_sentence_mask, = word_level_sentence_mask,
        self.char_level_sentence_mask = char_level_sentence_mask
        self.text_tag = text_tag


class TypeDataset(Dataset):

    def __init__(self, config, corpus_type):
        super(TypeDataset, self).__init__()
        self.config = config
        if corpus_type == "train":
            self.corpus_path = self.config.data.train_path
            self.guidance_path = self.config.data.guidance_train_path
        elif corpus_type == "dev":
            self.corpus_path = self.config.data.dev_path
            self.guidance_path = self.config.data.guidance_dev_path
        else:
            self.corpus_path = self.config.data.test_path
            self.guidance_path = self.config.data.guidance_test_path

        self.config.data.tags_path = os.path.join(self.config.data.data_dir, self.config.data.tags_path)
        self.config.data.out_tag_names_path = os.path.join(self.config.data.data_dir,
                                                           self.config.data.out_tag_names_path)

        self.config.data.word_vocab_path = os.path.join(self.config.data.data_dir, self.config.data.word_vocab_path)
        self.config.data.pos_tag_vocab_path = os.path.join(self.config.data.data_dir,
                                                           self.config.data.pos_tag_vocab_path)
        self.config.data.dep_tag_vocab_path = os.path.join(self.config.data.data_dir,
                                                           self.config.data.dep_tag_vocab_path)
        self.config.data.inp_tag_vocab_path = os.path.join(self.config.data.data_dir,
                                                           self.config.data.inp_tag_vocab_path)
        self.config.data.tag_emb_path = os.path.join(self.config.data.data_dir, self.config.data.tag_emb_path)

        # we prepare char/pattern level embeddings even if not training on it, to get input dimension etc. set
        assert self.config.use_char != "none" or self.config.pattern.use_pattern != "none", \
            "either char or pattern embeddings need to be used"

        if self.config.use_char != "none":
            print("dataset using char embeddings")
        if self.config.pattern.use_pattern != "none":
            print("dataset using pattern embeddings")

        if self.config.use_char == "lower":
            self.char_parser = LowercaseCharParser(max_word_len=self.config.max_word_len, include_special_chars=True,
                                                   post_padding=self.config.post_padding)
        else:
            self.char_parser = CharParser(max_word_len=self.config.max_word_len, include_special_chars=True,
                                          post_padding=self.config.post_padding)

        if self.config.pattern.use_pattern == "one-to-one":
            self.pattern_parser = OneToOnePatternParser(max_word_len=self.config.max_word_len,
                                                        include_special_chars=True,
                                                        post_padding=self.config.post_padding)
        elif self.config.pattern.use_pattern == "condensed":
            self.pattern_parser = WordCondensedPatternParser(max_word_len=self.config.max_word_len,
                                                             include_special_chars=True,
                                                             post_padding=self.config.post_padding,
                                                             retain_digits=self.config.pattern.retain_digits,
                                                             include_word_lengths=self.config.pattern.include_word_lengths)

        self.inp_dim = 0
        if self.config.use_char != "none":
            self.inp_dim += len(self.char_parser.vocab)
        if self.config.pattern.use_pattern != "none":
            self.inp_dim += len(self.pattern_parser.vocab)

        self.inp_tags = []
        self.out_tags = []
        self.out_tag_names = []
        self.parse_tags()

        self.word_vocab = []
        self.word_vocab_index = dict()
        self.pos_tag_vocab = []
        self.dep_tag_vocab = []
        self.parse_vocab()

        self.word_emb = []
        self.inp_tag_emb = []
        self.out_tag_emb = []
        self.tag_emb_dim = 0
        self.parse_embfile()

        self.text_sentences = []
        self.word_indexed_sentences = []
        self.char_indexed_sentences = []
        self.pattern_indexed_sentences = []
        self.type_indexed_sentences = []
        self.word_level_masks = []
        self.char_level_masks = []
        self.text_tags = []
        self.parse_dataset()

    def parse_tags(self):
        self.out_tags.append(self.config.pad_tag)
        with open(self.config.data.tags_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tags.append(line)

        # with open(self.config.data.out_tag_names_path, "r") as f:
        #     for line in f:
        #         line = line.strip()
        #         if line:
        #             self.out_tag_names.append(line)

        with open(self.config.data.inp_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line not in [self.config.pad_tag, self.config.none_tag]:
                    self.inp_tags.append(line)
                    assert line not in self.out_tags, "found overlapping tokens between inputs/output tag files"

    def parse_vocab(self):
        self.word_vocab.append(self.config.pad_tag)
        with open(self.config.data.word_vocab_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.word_vocab.append(line)
        self.word_vocab.append(self.config.unk_tag)

        for i, word in enumerate(self.word_vocab):
            self.word_vocab_index[word] = i

        self.pos_tag_vocab.append(self.config.pad_tag)
        with open(self.config.data.pos_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pos_tag_vocab.append(line)

        self.dep_tag_vocab.append(self.config.pad_tag)
        with open(self.config.data.dep_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.dep_tag_vocab.append(line)

    def parse_embfile(self):
        if self.config.data.word_emb_path and self.config.use_word == "glove":
            emb_dict = parse_emb_file(self.config.data.word_emb_path, has_header_line=False)
            emb_dict[self.config.unk_tag] = [0.0] * self.config.word_emb_dim
            emb_dict[self.config.pad_tag] = [0.0] * self.config.word_emb_dim
            self.word_emb = np.array([emb_dict[word] for word in self.word_vocab], dtype=np.float32)

        assert self.config.data.tag_emb_path, "tag embeddings input file always needs to be provided even if not being used"
        emb_dict = parse_emb_file(self.config.data.tag_emb_path, has_header_line=False)
        tag_emb_dim = 0
        for k in emb_dict.keys():
            tag_emb_dim = len(emb_dict[k])
            break
        self.out_tag_emb = np.array(self.read_tag_emb(self.out_tags, emb_dict, tag_emb_dim), dtype=np.float32)

        if self.config.use_tag_info == "pretrained":
            self.inp_tag_emb = np.array(self.read_tag_emb(self.inp_tags, emb_dict, tag_emb_dim), dtype=np.float32)
            self.tag_emb_dim = self.out_tag_emb.shape[1]
        else:
            self.tag_emb_dim = len(self.inp_tags)
            self.inp_tag_emb = np.identity(self.tag_emb_dim, dtype=np.float32)

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
                else:
                    raise ValueError("unexpected tag: {0}".format(tag))
        return tag_emb

    def parse_dataset(self):
        text_sentences = self.read_dataset(self.corpus_path)
        if self.config.use_tag_info != "none":
            guide_sentences = self.read_dataset(self.guidance_path)
            self.add_guidance(text_sentences, guide_sentences)
        for text_sentence in text_sentences:
            self.add_processed_sentence_tag(self.process_sentence_and_tag(text_sentence))

    def add_processed_sentence_tag(self, out):
        self.text_sentences.append(out.text_sentence)
        self.word_indexed_sentences.append(out.word_indexed_sentence)
        self.word_level_masks.append(out.word_level_sentence_mask)
        self.char_indexed_sentences.append(out.char_indexed_sentence)
        self.pattern_indexed_sentences.append(out.pattern_indexed_sentence)
        self.char_level_masks.append(out.char_level_sentence_mask)
        self.type_indexed_sentences.append(out.type_indexed_sentence)
        self.text_tags.append(out.text_tag)

    def read_dataset(self, file_path):
        text_sentences = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            text_sentence = []
            for row in reader:
                if len(row) == 0:
                    text_sentences.append(text_sentence)
                    text_sentence = []
                else:
                    if len(row) >= 3:
                        text_sentence.append(Token(start=0, text=row[0], pos_tag=row[1], dep_tag=row[2], tag=row[-1]))
                    else:
                        text_sentence.append(Token(start=0, text=row[0], tag=row[-1]))

        return text_sentences

    def add_guidance(self, text_sentences, guidance_sentences):
        if self.config.use_tag_info == "none":
            return
        assert len(text_sentences) == len(guidance_sentences), "guidance and input corpus need to have same sentences"
        for sent_index in range(len(text_sentences)):
            for token_index in range(len(text_sentences[sent_index])):
                token = text_sentences[sent_index][token_index]
                if token_index < len(guidance_sentences[sent_index]):
                    guide = guidance_sentences[sent_index][token_index]
                    assert token.text == guide.text, "token text mismatch at token {0} in sentence {1}" \
                        .format(token_index, sent_index)
                    token.guidance_tag = guide.tag
                else:
                    token.guidance_tag = self.config.none_tag

    def __len__(self):
        return len(self.text_sentences)

    def __getitem__(self, index):
        overall_char_indexed_sentence = []

        for word_index in range(len(self.text_sentences[index])):
            mappings = []
            if self.config.use_char != "none":
                indexed_sentence = self.char_indexed_sentences[index][word_index]
                mappings.append(self.char_parser.get_mapping(indexed_sentence))
            if self.config.pattern.use_pattern != "none":
                indexed_sentence = self.pattern_indexed_sentences[index][word_index]
                mappings.append(self.pattern_parser.get_mapping(indexed_sentence))
            overall_char_indexed_sentence.append(np.hstack(mappings))

        text_sentence = self.text_sentences[index]
        overall_char_indexed_sentence = np.array(overall_char_indexed_sentence, dtype=np.float32)
        word_indexed_sentence = self.word_indexed_sentences[index]
        type_indexed_sentence = self.type_indexed_sentences[index]
        word_level_mask = self.word_level_masks[index]
        char_level_mask = self.char_level_masks[index]
        indexed_tag = self.get_indexed_tag(self.text_tags[index])

        return \
            text_sentence, \
            word_indexed_sentence, \
            overall_char_indexed_sentence, \
            type_indexed_sentence, \
            word_level_mask, \
            char_level_mask, \
            indexed_tag

    def get_query_given_tokens(self, text_sentence):
        token_sentence = [
            Token(start=0, text=text, pos_tag=self.config.pad_tag, dep_tag=self.config.pad_tag, tag=self.config.pad_tag)
            for text in text_sentence]
        out = self.process_sentence_and_tag(token_sentence)
        indexed_tag = self.get_indexed_tag(out.text_tag)

        return \
            out.text_sentence, \
            out.word_indexed_sentence, \
            self.get_char_indexed_sentence(out), \
            out.type_indexed_sentence, \
            out.word_level_sentence_mask, \
            out.char_level_sentence_mask, \
            indexed_tag

    def get_indexed_tag(self, text_tag):
        indexed_tag = []
        for curr_tag in text_tag:
            if curr_tag in self.out_tags:
                indexed_tag.append(self.out_tags.index(curr_tag))
            else:
                indexed_tag.append(self.out_tags.index(self.config.none_tag))
        return np.array(indexed_tag, dtype=np.int64)

    def get_char_indexed_sentence(self, out):
        overall_char_indexed_sentence = []
        for word_index in range(len(out.text_sentence)):
            mappings = []
            if self.config.use_char != "none":
                mappings.append(self.char_parser.get_mapping(out.char_indexed_sentence[word_index]))
            if self.config.pattern.use_pattern != "none":
                mappings.append(self.pattern_parser.get_mapping(out.pattern_indexed_sentence[word_index]))
            overall_char_indexed_sentence.append(np.hstack(mappings))
        overall_char_indexed_sentence = np.array(overall_char_indexed_sentence, dtype=np.float32)
        return overall_char_indexed_sentence

    def process_sentence_and_tag(self, token_sentence):
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
        char_indexed_sentence = []
        pattern_indexed_sentence = []
        type_indexed_sentence = []
        char_level_sentence_mask = []
        for index in range(len(token_sentence)):
            if self.config.use_tag_info != "none":
                if token_sentence[index].guidance_tag in self.inp_tags:
                    tag_vec = self.inp_tag_emb[self.inp_tags.index(token_sentence[index].guidance_tag)]
                else:
                    tag_vec = np.zeros(self.tag_emb_dim, dtype=np.float32)
                type_indexed_sentence.append(tag_vec)

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

            char_indexed_word, char_level_word_mask = self.char_parser.get_indexed_text(word.text)
            char_level_sentence_mask.append(char_level_word_mask)
            if self.config.use_char != "none":
                char_indexed_sentence.append(char_indexed_word)
            if self.config.pattern.use_pattern != "none":
                pattern_indexed_word, char_level_word_mask = self.pattern_parser.get_indexed_text(word.text)
                pattern_indexed_sentence.append(pattern_indexed_word)

        word_indexed_sentence = np.array(word_indexed_sentence, dtype=np.int64)
        type_indexed_sentence = np.array(type_indexed_sentence)
        char_level_sentence_mask = np.array(char_level_sentence_mask)
        word_level_sentence_mask = np.array(word_level_sentence_mask)

        text_sentence = [token.text for token in token_sentence]
        text_tag = [token.tag for token in token_sentence]

        return ProcessedSentenceAndTag(text_sentence, char_indexed_sentence, pattern_indexed_sentence,
                                       word_indexed_sentence, type_indexed_sentence, word_level_sentence_mask,
                                       char_level_sentence_mask, text_tag)
