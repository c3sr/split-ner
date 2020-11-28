import csv
import os

import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer

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


class BertToken(Token):
    def __init__(self, start, text, bert_id, tag, pos_tag=None, dep_tag=None):
        super(BertToken, self).__init__(start, text, tag, pos_tag, dep_tag)
        self.bert_id = bert_id

    def __str__(self):
        return "({0}, {1}, {2}, {3}, {4}, {5})".format(self.start, self.text, self.bert_id, self.tag, self.pos_tag,
                                                       self.dep_tag)

    def __repr__(self):
        return "({0}, {1}, {2}, {3}, {4}, {5})".format(self.start, self.text, self.bert_id, self.tag, self.pos_tag,
                                                       self.dep_tag)


class TypeDataset(Dataset):

    def __init__(self, corpus_path, config):
        super(TypeDataset, self).__init__()
        self.config = config
        self.corpus_path = corpus_path

        self.punct_tag = "punct"
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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

        with open(self.config.data.out_tag_names_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tag_names.append(line)

        with open(self.config.data.inp_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
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
        text_sentences, text_tags = self.read_dataset()
        for text_sentence, text_tag in zip(text_sentences, text_tags):
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

    def process_bert(self, text_sentence):
        # use self.seq_length
        text = [token.text for token in text_sentence]
        bert_batch_tokens = self.bert_tokenizer(text, is_pretokenized=True,
                                                padding="max_length", max_length=self.config.max_seq_len)["input_ids"]
        # run for loop for individual tokens and duplicate tokens
        # TODO: Figure out word-level masking, padding, B/I-tag copy to final
        # TODO: then commit and run on server
        sent = [[token.text] for token in text_sentence]
        bert_sent_tokens = self.bert_tokenizer(sent, is_pretokenized=True)["input_ids"]
        word_ending = [max(1, len(bert_word_tokens) - 2) for bert_word_tokens in bert_sent_tokens]
        bert_tokens_text = [tok.replace("##", "") for tok in
                            self.bert_tokenizer.convert_ids_to_tokens(bert_batch_tokens)]
        new_text_sentence = [
            BertToken(start=0, text=bert_tokens_text[0], bert_id=bert_batch_tokens[0], tag=self.config.none_tag,
                      pos_tag=self.config.none_tag, dep_tag=self.punct_tag)]
        cnt = 1
        for i in range(len(text_sentence)):
            token = text_sentence[i]
            tag = token.tag
            new_text_sentence.append(BertToken(start=token.start, text=bert_tokens_text[cnt],
                                               bert_id=bert_batch_tokens[cnt], tag=tag, pos_tag=token.pos_tag,
                                               dep_tag=token.dep_tag))
            cnt += 1
            if tag[:2] == "B-":
                tag = "I-" + tag[2:]
            for j in range(1, word_ending[i]):
                new_token = BertToken(start=token.start, text=bert_tokens_text[cnt], bert_id=bert_batch_tokens[cnt],
                                      tag=tag, pos_tag=token.pos_tag, dep_tag=token.dep_tag)
                new_text_sentence.append(new_token)
                cnt += 1

        return new_text_sentence

    def read_dataset(self):
        text_sentences = []
        with open(self.corpus_path, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            text_sentence = []
            text_tag = []
            for row in reader:
                if len(row) == 0:
                    text_sentences.append(self.process_bert(text_sentence))
                    text_sentence = []
                    text_tag = []
                else:
                    if len(row) == 2:
                        # also have word_level_sentence_mask here (in each BertToken) [pre/post-padding]
                        text_sentence.append(Token(start=0, text=row[0], tag=row[-1]))
                    else:
                        text_sentence.append(Token(start=0, text=row[0], pos_tag=row[1], dep_tag=row[2], tag=row[-1]))
                    text_tag.append(row[-1])

        return text_sentences

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
        text_tag = []
        token_sentence = [
            Token(start=0, text=text, pos_tag=self.config.pad_tag, dep_tag=self.config.pad_tag, tag=self.config.pad_tag)
            for
            text in text_sentence]
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
        return np.array(indexed_tag)

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
                    BertToken(start=0, text="", tag=self.config.pad_tag, pos_tag=self.config.none_tag,
                              dep_tag=self.config.punct_tag,
                              bert_id=0)] * (
                                         self.config.max_seq_len - len(token_sentence))
            else:
                word_level_sentence_mask = [0] * (self.config.max_seq_len - len(token_sentence)) + [1] * len(
                    token_sentence)
                token_sentence = [BertToken(start=0, text="", tag=self.config.none_tag, pos_tag=self.config.punct_tag,
                                            dep_tag=self.config.pad_tag, bert_id=0)] * (
                                         self.config.max_seq_len - len(token_sentence)) + token_sentence
        word_indexed_sentence = []
        char_indexed_sentence = []
        pattern_indexed_sentence = []
        type_indexed_sentence = []
        char_level_sentence_mask = []
        for index in range(len(token_sentence)):
            if self.config.use_tag_info == "window":
                tag_context = []
                for i in range(max(0, index - self.config.window_size), index):
                    if token_sentence[i].tag in self.inp_tags:
                        tag_vec = self.inp_tag_emb[self.inp_tags.index(token_sentence[i].tag)]
                    else:
                        tag_vec = np.zeros(self.tag_emb_dim, dtype=np.float32)
                    tag_context.append(tag_vec)
                    # not taking the output tag info of the current tag (since, it might make the model do trivial copy
                    # from input to output)
                for i in range(index + 1, min(index + self.config.window_size + 1, len(token_sentence))):
                    if token_sentence[i].tag in self.inp_tags:
                        tag_vec = self.inp_tag_emb[self.inp_tags.index(token_sentence[i].tag)]
                    else:
                        tag_vec = np.zeros(self.tag_emb_dim, dtype=np.float32)
                    tag_context.append(tag_vec)
                tag_context = np.vstack(tag_context)
                tag_context = np.amax(tag_context, axis=0)
                type_indexed_sentence.append(tag_context)
            else:  # context: none / self
                if token_sentence[index].tag in self.inp_tags:
                    tag_vec = self.inp_tag_emb[self.inp_tags.index(token_sentence[index].tag)]
                else:
                    tag_vec = np.zeros(self.tag_emb_dim, dtype=np.float32)
                type_indexed_sentence.append(tag_vec)

            word = token_sentence[index]
            word_index = [word.bert_id, self.pos_tag_vocab.index(word.pos_tag),
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
