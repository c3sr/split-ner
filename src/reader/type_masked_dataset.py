import numpy as np

from src.reader.type_dataset import TypeDataset


class TypeMaskedDataset(TypeDataset):

    def __init__(self, corpus_path, word_vocab_path, out_tag_vocab_path, inp_tag_vocab_path, mask_tag_vocab_path,
                 pos_tag_vocab_path,
                 dep_tag_vocab_path,
                 word_emb_path=None, tag_emb_path=None, use_char="lower", use_pattern="condensed", use_word="glove",
                 include_word_lengths=False, retain_digits=False, pad_tag="<PAD>", none_tag="O", unk_tag="<UNK>",
                 word_emb_dim=50, max_word_len=20, max_seq_len=20, post_padding=True, use_tag_info="self",
                 window_size=5):

        self.mask_tag_vocab_path = mask_tag_vocab_path
        self.mask_tags = []

        super(TypeMaskedDataset, self).__init__(corpus_path, word_vocab_path, out_tag_vocab_path, inp_tag_vocab_path,
                                                pos_tag_vocab_path, dep_tag_vocab_path,
                                                word_emb_path, tag_emb_path, use_char, use_pattern, use_word,
                                                include_word_lengths, retain_digits, pad_tag, none_tag, unk_tag,
                                                word_emb_dim, max_word_len, max_seq_len, post_padding, use_tag_info,
                                                window_size)

    def parse_tags(self):
        with open(self.out_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tags.append(line)

        with open(self.inp_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.inp_tags.append(line)
                    assert line not in self.out_tags, "found overlapping tokens between inputs/output tag files"

        self.mask_tags.append(self.pad_tag)
        with open(self.mask_tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    assert line not in self.out_tags, "Error: masked tag {0} found in out tags too.".format(line)
                    self.mask_tags.append(line)

    def __getitem__(self, index):
        tag_level_mask = self.get_tag_level_sentence_mask(self.text_tags[index])

        text_sentence, \
        word_indexed_sentence, \
        overall_char_indexed_sentence, \
        type_indexed_sentence, \
        word_level_mask, \
        char_level_mask, \
        indexed_tag = TypeDataset.__getitem__(self, index)

        return \
            text_sentence, \
            word_indexed_sentence, \
            overall_char_indexed_sentence, \
            type_indexed_sentence, \
            word_level_mask, \
            char_level_mask, \
            tag_level_mask, \
            indexed_tag

    def get_query_given_tokens(self, text_sentence):
        text_sentence, \
        word_indexed_sentence, \
        overall_char_indexed_sentence, \
        type_indexed_sentence, \
        word_level_mask, \
        char_level_mask, \
        indexed_tag = TypeDataset.get_query_given_tokens(self, text_sentence)

        tag_level_mask = np.ones_like(indexed_tag)

        return \
            text_sentence, \
            word_indexed_sentence, \
            overall_char_indexed_sentence, \
            type_indexed_sentence, \
            word_level_mask, \
            char_level_mask, \
            tag_level_mask, \
            indexed_tag

    def get_indexed_tag(self, text_tag):
        indexed_tag = []
        for curr_tag in text_tag:
            tag = curr_tag
            if tag not in self.out_tags:
                tag = self.none_tag
            if tag in self.mask_tags:
                indexed_tag.append(0)  # could be any dummy but valid index, we would not use it
            else:
                indexed_tag.append(self.out_tags.index(tag))
        return np.array(indexed_tag)

    def get_tag_level_sentence_mask(self, text_tag):
        tag_level_mask = []
        for curr_tag in text_tag:
            tag = curr_tag
            if tag not in self.out_tags:
                tag = self.none_tag
            if tag in self.mask_tags:
                tag_level_mask.append(0)
            else:
                tag_level_mask.append(1)
        return np.array(tag_level_mask)
