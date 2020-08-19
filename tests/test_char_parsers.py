from unittest import TestCase

from src.utils.char_parsers import WordCondensedPatternParser


class TestWordCondensedPatternParser(TestCase):

    def test_get_encoded_text(self):
        no_digits_parser = WordCondensedPatternParser(include_word_lengths=False, retain_digits=False)
        assert no_digits_parser.get_encoded_text("c.2708_2711delTTAC")[0].startswith("l.dddd_ddddm ")
        assert no_digits_parser.get_encoded_text("c4.5")[0].startswith("ld.d ")
        assert no_digits_parser.get_encoded_text("c 4.5")[0].startswith("ld.d ")
        assert no_digits_parser.get_encoded_text("ABCabc")[0].startswith("m ")
        assert no_digits_parser.get_encoded_text("Abcd-EF-Gh")[0].startswith("i-u-i ")
        assert no_digits_parser.get_encoded_text("")[0].startswith(" ")
        assert no_digits_parser.get_encoded_text("EX123")[0].startswith("uddd")
        assert no_digits_parser.get_encoded_text("123")[0].startswith("ddd")

        word_lengths_parser = WordCondensedPatternParser(include_word_lengths=True, retain_digits=False)
        assert word_lengths_parser.get_encoded_text("c.2708_2711delTTAC")[0].startswith("l1.2708_2711m7 ")

        digit_and_lengths_parser = WordCondensedPatternParser(include_word_lengths=True, retain_digits=True)
        assert digit_and_lengths_parser.get_encoded_text("c.2708_2711delTTAC")[0].startswith("l1.2708_2711m7 ")

        retain_digits_parser = WordCondensedPatternParser(include_word_lengths=False, retain_digits=True)
        assert retain_digits_parser.get_encoded_text("c.2708_2711delTTAC")[0].startswith("l.2708_2711m ")
