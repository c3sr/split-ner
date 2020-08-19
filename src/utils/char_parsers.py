import re

import numpy as np


class BaseParser:

    def __init__(self, vocab, max_word_len=20, include_special_chars=True, post_padding=True):
        self.max_word_len = max_word_len
        self.post_padding = post_padding
        self.vocab = vocab
        if include_special_chars:
            vocab += list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        self.mapping = np.identity(len(self.vocab), dtype=np.float32)

    def get_encoded_text(self, text):
        return self.encode(text)

    def encode(self, text):
        # ignoring chars which are out of vocab
        chars = [ch for ch in list(text) if ch in self.vocab]
        if len(chars) > self.max_word_len:
            processed_chars = chars[:self.max_word_len]
            mask = [1] * self.max_word_len
        else:
            if self.post_padding:
                # post-padding with spaces
                processed_chars = chars + [" "] * (self.max_word_len - len(chars))
                mask = [1] * len(chars) + [0] * (self.max_word_len - len(chars))
            else:
                # pre-padding with spaces
                processed_chars = [" "] * (self.max_word_len - len(chars)) + chars
                mask = [0] * (self.max_word_len - len(chars)) + [1] * len(chars)
        encoded_text = "".join(processed_chars)
        return encoded_text, mask

    def get_indexed_text(self, text):
        encoded_text, mask = self.get_encoded_text(text)
        indexed_text = [self.vocab.index(encoded_text[i]) if mask[i] == 1 else -1 for i in range(len(mask))]
        return indexed_text, mask

    def get_mapping(self, indexed_text):
        return [self.mapping[i].flatten() if i >= 0 else np.zeros((len(self.vocab)), dtype=np.float32) for i in
                indexed_text]


class LowercaseCharParser(BaseParser):

    def __init__(self, max_word_len=20, include_special_chars=True, post_padding=True):
        vocab = list("abcdefghijklmnopqrstuvwxyz0123456789")
        super(LowercaseCharParser, self).__init__(vocab=vocab, include_special_chars=include_special_chars,
                                                  max_word_len=max_word_len, post_padding=post_padding)

    def get_encoded_text(self, text):
        return self.encode(text.lower())


class CharParser(BaseParser):

    def __init__(self, max_word_len=20, include_special_chars=True, post_padding=True):
        vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        super(CharParser, self).__init__(vocab=vocab, include_special_chars=include_special_chars,
                                         max_word_len=max_word_len, post_padding=post_padding)


class OneToOnePatternParser(BaseParser):

    def __init__(self, max_word_len=20, include_special_chars=True, post_padding=True):
        """
        l: lower-case alpha
        u: upper-case alpha
        d: digit
        special symbols retained as it is
        """
        vocab = list("lud")
        super(OneToOnePatternParser, self).__init__(vocab=vocab, include_special_chars=include_special_chars,
                                                    max_word_len=max_word_len, post_padding=post_padding)

    def get_encoded_text(self, text):
        text = re.sub(r"[a-z]", "l", text)
        text = re.sub(r"[A-Z]", "u", text)
        text = re.sub(r"[0-9]", "d", text)
        return self.encode(text)


class WordCondensedPatternParser(BaseParser):
    def __init__(self, max_word_len=20, include_special_chars=True, post_padding=True, include_word_lengths=False,
                 retain_digits=False):
        """
        l: lower-case word
        u: upper-case word
        m: mixed-case word
        i: initial letter (upper-case) followed by lower-case
        d: digit (optional)
        special symbols retained as it is
        """
        self.retain_digits = retain_digits
        self.include_word_lengths = include_word_lengths

        vocab = list("lumi")
        if self.retain_digits or self.include_word_lengths:
            vocab.extend(list("0123456789"))
        else:
            vocab.append("d")
        super(WordCondensedPatternParser, self).__init__(vocab=vocab, include_special_chars=include_special_chars,
                                                         max_word_len=max_word_len, post_padding=post_padding)

    def get_encoded_text(self, text):
        mod_text = ""
        start = 0
        for m in re.finditer(r"[A-Za-z]+", text):
            mod_text += text[start: m.start()]
            sub = m.group()
            if sub.islower():
                mod_text += "l"
            elif sub.isupper():
                mod_text += "u"
            elif sub[0].isupper() and sub[1:].islower():
                mod_text += "i"
            else:
                mod_text += "m"

            if self.include_word_lengths:
                mod_text += str(len(sub))
            start = m.end()
        mod_text += text[start:]
        if not (self.retain_digits or self.include_word_lengths):
            mod_text = re.sub(r"[0-9]", "d", mod_text)

        return self.encode(mod_text)
