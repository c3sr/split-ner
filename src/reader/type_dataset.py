import csv

import numpy as np
from torch.utils.data import Dataset

from src.datasets.genia import Token
from src.utils.char_parsers import LowercaseCharParser, CharParser, OneToOnePatternParser, WordCondensedPatternParser
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

    def __init__(self, corpus_path, word_vocab_path, out_tag_vocab_path, out_tag_names_path, inp_tag_vocab_path,
                 pos_tag_vocab_path, dep_tag_vocab_path, word_emb_path=None, tag_emb_path=None, use_char="lower",
                 use_pattern="condensed", use_word="glove", include_word_lengths=False, retain_digits=False,
                 pad_tag="<PAD>", none_tag="O", unk_tag="<UNK>", word_emb_dim=50, max_word_len=20, max_seq_len=20,
                 post_padding=True, use_tag_info="self", window_size=5):
        super(TypeDataset, self).__init__()
        self.corpus_path = corpus_path
        self.word_vocab_path = word_vocab_path
        self.pos_tag_vocab_path = pos_tag_vocab_path
        self.dep_tag_vocab_path = dep_tag_vocab_path
        self.out_tag_vocab_path = out_tag_vocab_path
        self.out_tag_names_path = out_tag_names_path
        self.inp_tag_vocab_path = inp_tag_vocab_path
        self.word_emb_path = word_emb_path
        self.tag_emb_path = tag_emb_path
        self.use_char = use_char
        self.use_pattern = use_pattern
        self.use_word = use_word
        self.pad_tag = pad_tag
        self.unk_tag = unk_tag
        self.none_tag = none_tag
        self.word_emb_dim = word_emb_dim
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len
        self.post_padding = post_padding
        self.use_tag_info = use_tag_info
        self.window_size = window_size

        # we prepare char/pattern level embeddings even if not training on it, to get input dimension etc. set
        assert self.use_char != "none" or self.use_pattern != "none", \
            "either char or pattern embeddings need to be used"

        if self.use_char != "none":
            print("dataset using char embeddings")
        if self.use_pattern != "none":
            print("dataset using pattern embeddings")

        if self.use_char == "lower":
            self.char_parser = LowercaseCharParser(max_word_len=self.max_word_len, include_special_chars=True,
                                                   post_padding=post_padding)
        else:
            self.char_parser = CharParser(max_word_len=self.max_word_len, include_special_chars=True,
                                          post_padding=post_padding)

        if self.use_pattern == "one-to-one":
            self.pattern_parser = OneToOnePatternParser(max_word_len=self.max_word_len, include_special_chars=True,
                                                        post_padding=post_padding)
        elif self.use_pattern == "condensed":
            self.pattern_parser = WordCondensedPatternParser(max_word_len=self.max_word_len, include_special_chars=True,
                                                             post_padding=post_padding, retain_digits=retain_digits,
                                                             include_word_lengths=include_word_lengths)

        self.inp_dim = 0
        if self.use_char != "none":
            self.inp_dim += len(self.char_parser.vocab)
        if self.use_pattern != "none":
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
        self.out_tags.append(self.pad_tag)
        with open(self.out_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tags.append(line)

        with open(self.out_tag_names_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tag_names.append(line)

        with open(self.inp_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.inp_tags.append(line)
                    assert line not in self.out_tags, "found overlapping tokens between inputs/output tag files"

    def parse_vocab(self):
        self.word_vocab.append(self.pad_tag)
        with open(self.word_vocab_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.word_vocab.append(line)
        self.word_vocab.append(self.unk_tag)

        for i, word in enumerate(self.word_vocab):
            self.word_vocab_index[word] = i

        self.pos_tag_vocab.append(self.pad_tag)
        with open(self.pos_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pos_tag_vocab.append(line)

        self.dep_tag_vocab.append(self.pad_tag)
        with open(self.dep_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.dep_tag_vocab.append(line)

    def parse_embfile(self):
        if self.word_emb_path and self.use_word == "glove":
            emb_dict = parse_emb_file(self.word_emb_path, has_header_line=False)
            emb_dict[self.unk_tag] = [0.0] * self.word_emb_dim
            emb_dict[self.pad_tag] = [0.0] * self.word_emb_dim
            self.word_emb = np.array([emb_dict[word] for word in self.word_vocab], dtype=np.float32)

        assert self.tag_emb_path, "tag embeddings input file always needs to be provided even if not being used"
        emb_dict = parse_emb_file(self.tag_emb_path, has_header_line=False)
        tag_emb_dim = 0
        for k in emb_dict.keys():
            tag_emb_dim = len(emb_dict[k])
            break
        self.out_tag_emb = np.array(self.read_tag_emb(self.out_tags, emb_dict, tag_emb_dim), dtype=np.float32)

        if self.use_tag_info == "pretrained":
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

                if tag == self.none_tag:
                    tag_emb.append(main_vec + [1.0, 0.0])
                elif tag == self.pad_tag:
                    tag_emb.append(main_vec + [0.0, 1.0])
                else:
                    raise ValueError("unexpected tag: {0}".format(tag))
        return tag_emb

    def parse_dataset(self):
        text_sentences, text_tags = self.read_dataset()
        for text_sentence, text_tag in zip(text_sentences, text_tags):
            self.add_processed_sentence_tag(self.process_sentence_and_tag(text_sentence, text_tag))

    def add_processed_sentence_tag(self, out):
        self.text_sentences.append(out.text_sentence)
        self.word_indexed_sentences.append(out.word_indexed_sentence)
        self.word_level_masks.append(out.word_level_sentence_mask)
        self.char_indexed_sentences.append(out.char_indexed_sentence)
        self.pattern_indexed_sentences.append(out.pattern_indexed_sentence)
        self.char_level_masks.append(out.char_level_sentence_mask)
        self.type_indexed_sentences.append(out.type_indexed_sentence)
        self.text_tags.append(out.text_tag)

    def read_dataset(self):
        text_sentences = []
        text_tags = []
        with open(self.corpus_path, "r") as f:
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
                    if len(row) == 2:
                        text_sentence.append(Token(start=0, text=row[0], tag=row[-1]))
                    else:
                        text_sentence.append(Token(start=0, text=row[0], pos_tag=row[1], dep_tag=row[2], tag=row[-1]))
                    text_tag.append(row[-1])

        return text_sentences, text_tags

    def __len__(self):
        return len(self.text_sentences)

    def __getitem__(self, index):
        overall_char_indexed_sentence = []

        for word_index in range(len(self.text_sentences[index])):
            mappings = []
            if self.use_char != "none":
                indexed_sentence = self.char_indexed_sentences[index][word_index]
                mappings.append(self.char_parser.get_mapping(indexed_sentence))
            if self.use_pattern != "none":
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
        text_tag = []
        token_sentence = [Token(start=0, text=text, pos_tag=self.pad_tag, dep_tag=self.pad_tag, tag=self.pad_tag) for
                          text in text_sentence]
        out = self.process_sentence_and_tag(token_sentence, text_tag)
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
                indexed_tag.append(self.out_tags.index(self.none_tag))
        return np.array(indexed_tag)

    def get_char_indexed_sentence(self, out):
        overall_char_indexed_sentence = []
        for word_index in range(len(out.text_sentence)):
            mappings = []
            if self.use_char != "none":
                mappings.append(self.char_parser.get_mapping(out.char_indexed_sentence[word_index]))
            if self.use_pattern != "none":
                mappings.append(self.pattern_parser.get_mapping(out.pattern_indexed_sentence[word_index]))
            overall_char_indexed_sentence.append(np.hstack(mappings))
        overall_char_indexed_sentence = np.array(overall_char_indexed_sentence, dtype=np.float32)
        return overall_char_indexed_sentence

    def process_sentence_and_tag(self, token_sentence, text_tag):
        if len(token_sentence) > self.max_seq_len:
            word_level_sentence_mask = [1] * self.max_seq_len
            token_sentence = token_sentence[:self.max_seq_len]
            text_tag = text_tag[:self.max_seq_len]
        else:
            if self.post_padding:
                word_level_sentence_mask = [1] * len(token_sentence) + [0] * (self.max_seq_len - len(token_sentence))
                token_sentence = token_sentence + [
                    Token(start=0, text="", tag=self.pad_tag, pos_tag=self.pad_tag, dep_tag=self.pad_tag)] * (
                                         self.max_seq_len - len(token_sentence))
                text_tag = text_tag + [self.pad_tag] * (self.max_seq_len - len(text_tag))
            else:
                word_level_sentence_mask = [0] * (self.max_seq_len - len(token_sentence)) + [1] * len(token_sentence)
                token_sentence = [Token(start=0, text="", tag=self.pad_tag, pos_tag=self.pad_tag,
                                        dep_tag=self.pad_tag)] * (
                                         self.max_seq_len - len(token_sentence)) + token_sentence
                text_tag = [self.pad_tag] * (self.max_seq_len - len(text_tag)) + text_tag
        word_indexed_sentence = []
        char_indexed_sentence = []
        pattern_indexed_sentence = []
        type_indexed_sentence = []
        char_level_sentence_mask = []
        for index in range(len(token_sentence)):
            if self.use_tag_info == "window":
                tag_context = []
                for i in range(max(0, index - self.window_size), index):
                    if text_tag[i] in self.inp_tags:
                        tag_vec = self.inp_tag_emb[self.inp_tags.index(text_tag[i])]
                    else:
                        tag_vec = np.zeros(self.tag_emb_dim, dtype=np.float32)
                    tag_context.append(tag_vec)
                    # not taking the output tag info of the current tag (since, it might make the model do trivial copy
                    # from input to output)
                for i in range(index + 1, min(index + self.window_size + 1, len(text_tag))):
                    if text_tag[i] in self.inp_tags:
                        tag_vec = self.inp_tag_emb[self.inp_tags.index(text_tag[i])]
                    else:
                        tag_vec = np.zeros(self.tag_emb_dim, dtype=np.float32)
                    tag_context.append(tag_vec)
                tag_context = np.vstack(tag_context)
                tag_context = np.amax(tag_context, axis=0)
                type_indexed_sentence.append(tag_context)
            else:  # context: none / self
                if text_tag[index] in self.inp_tags:
                    tag_vec = self.inp_tag_emb[self.inp_tags.index(text_tag[index])]
                else:
                    tag_vec = np.zeros(self.tag_emb_dim, dtype=np.float32)
                type_indexed_sentence.append(tag_vec)

            word = token_sentence[index]
            if word_level_sentence_mask[index] == 0:
                word_index = [self.word_vocab_index[self.pad_tag], self.pos_tag_vocab.index(word.pos_tag),
                              self.dep_tag_vocab.index(word.dep_tag)]
                word_indexed_sentence.append(word_index)  # pad tag
            else:
                lw = word.text.lower()
                if lw not in self.word_vocab_index:
                    lw = self.unk_tag  # unknown tag
                word_index = [self.word_vocab_index[lw], self.pos_tag_vocab.index(word.pos_tag),
                              self.dep_tag_vocab.index(word.dep_tag)]
                word_indexed_sentence.append(word_index)

            char_indexed_word, char_level_word_mask = self.char_parser.get_indexed_text(word.text)
            char_level_sentence_mask.append(char_level_word_mask)
            if self.use_char != "none":
                char_indexed_sentence.append(char_indexed_word)
            if self.use_pattern != "none":
                pattern_indexed_word, char_level_word_mask = self.pattern_parser.get_indexed_text(word.text)
                pattern_indexed_sentence.append(pattern_indexed_word)

        word_indexed_sentence = np.array(word_indexed_sentence)
        type_indexed_sentence = np.array(type_indexed_sentence)
        char_level_sentence_mask = np.array(char_level_sentence_mask)
        word_level_sentence_mask = np.array(word_level_sentence_mask)

        text_sentence = [token.text for token in token_sentence]

        return ProcessedSentenceAndTag(text_sentence, char_indexed_sentence, pattern_indexed_sentence,
                                       word_indexed_sentence, type_indexed_sentence, word_level_sentence_mask,
                                       char_level_sentence_mask, text_tag)
