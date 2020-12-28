import argparse
import os
import re

import spacy
from sklearn.model_selection import train_test_split

from src.datasets.general_tsv import generate_corpus_files
from src.utils.corpus_utils import create_vocab_from_embeddings
from src.utils.dataset_utils import Token

nlp = spacy.load("en_core_sci_sm")


class CorpusSample:

    def __init__(self, text, spans):
        self.text = text
        self.spans = spans


def read_corpus(corpus_path):
    title_prefix_pat = re.compile(r"^\d+\|t\|(.*)$")
    abstract_prefix_pat = re.compile(r"^\d+\|a\|(.*)$")

    corpus = []
    with open(corpus_path, "r") as f:
        text = ""
        spans = []
        index = 0
        for line in f:
            line = line.strip()
            if not line:
                corpus.append(CorpusSample(text, spans))
                text = ""
                spans = []
                index = 0
                continue
            if index == 0:
                text += title_prefix_pat.match(line).group(1)
            elif index == 1:
                text += " " + abstract_prefix_pat.match(line).group(1)
            else:
                s = line.split("\t")
                start = int(s[1])
                end = int(s[2])
                tag = s[4]
                spans.append(Token(start, text[start: end], tag))
            index += 1
    return corpus


def process_corpus(corpus_path):
    corpus = read_corpus(corpus_path)
    pat = r"EntityReplace"

    tagged_corpus = []
    for sample in corpus:
        text = replace_mentions_with_pattern(sample, pat)
        sentences = [s + "." for s in text.split(". ")]
        sentences = replace_pattern_with_mentions(sample, sentences, pat)
        tagged_corpus.extend(tokenize_and_tag(sample, sentences))
    return tagged_corpus


def tokenize_and_tag(sample, sentences):
    res = []
    start = 0
    for sentence in sentences:
        tokens = nlp(sentence)
        token_tags = []
        for token in tokens:
            token_start = start + token.idx
            token_end = token_start + len(token)
            token_tag = "O"
            for span in sample.spans:
                if span.start <= token_start < span.start + len(span.text):
                    token_tag = span.tag
                    break
                if span.start > token_end:
                    break
            pos_tag = token.tag_ if len(token.tag_) > 0 else "O"
            dep_tag = token.dep_ if len(token.dep_) > 0 else "O"
            token_tags.append(Token(start=0, text=token.text, tag=token_tag, pos_tag=pos_tag, dep_tag=dep_tag))
        start += len(sentence) + 1
        res.append(token_tags)

    for sent_tokens in res:
        prev_entity = "O"
        for token in sent_tokens:
            if token.tag == "O":
                prev_entity = "O"
                continue
            if token.tag == prev_entity:
                token.tag = "I-{0}".format(prev_entity)
            else:
                prev_entity = token.tag
                token.tag = "B-{0}".format(prev_entity)
    return res


def replace_pattern_with_mentions(sample, sentences, pat):
    index = 0
    res = []
    for sentence in sentences:
        while True:
            m = re.search(pat, sentence)
            if not m:
                break
            span_start = sample.spans[index].start
            span_end = span_start + len(sample.spans[index].text)
            sentence = sentence[:m.start()] + sample.text[span_start: span_end] + sentence[m.end():]
            index += 1
        res.append(sentence)
    return res


def replace_mentions_with_pattern(sample, pat):
    text = sample.text
    for s in reversed(sample.spans):
        span_end = s.start + len(s.text)
        text = text[:s.start] + pat + text[span_end:]
    return text


def tag_to_text_tmvar(tag):
    if tag == "DNAMutation":
        return "DNA-level mutation"
    if tag == "ProteinMutation":
        return "protein-level mutation"
    return "single nucleotide polymorphism level mutation"  # tag == "SNP"


def main(args):
    corpus_non_test = process_corpus(os.path.join(args.dir, args.inp_train))
    corpus_train, corpus_dev = train_test_split(corpus_non_test, test_size=0.25, random_state=args.seed)
    corpus_test = process_corpus(os.path.join(args.dir, args.inp_test))
    corpus = corpus_train + corpus_dev + corpus_test

    # utility to read existing corpus
    # corpus = read_tsv_corpus(os.path.join(args.dir, "corpus.tsv"))
    # corpus_train = read_tsv_corpus(os.path.join(args.dir, "train.tsv"))
    # corpus_dev = read_tsv_corpus(os.path.join(args.dir, "dev.tsv"))
    # corpus_test = read_tsv_corpus(os.path.join(args.dir, "test.tsv"))

    generate_corpus_files(args.dir, corpus, corpus_train, corpus_dev, corpus_test, tag_to_text_fn=tag_to_text_tmvar)
    create_vocab_from_embeddings(embpath=args.glove_emb_path, vocabpath=os.path.join(args.dir, "glove_vocab.txt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TMVar corpus parser (convert to TSV)")
    ap.add_argument("--dir", type=str, default="../../data/tmVarCorpus",
                    help="corpus dir (Default: '../../data/tmVarCorpus')")
    ap.add_argument("--inp_train", type=str, default="train.PubTator.txt",
                    help="path to input train file (.txt), relative to corpus dir (Default: 'train.PubTator.txt')")
    ap.add_argument("--inp_test", type=str, default="test.PubTator.txt",
                    help="path to input test file (.txt) relative to corpus dir (Default: 'test.PubTator.txt')")
    ap.add_argument("--glove_emb_path", type=str, default="../../../../Embeddings/glove.6B.50d.txt",
                    help="glove embeddings file path (Default: '../../../../Embeddings/glove.6B.50d.txt')")
    ap.add_argument("-s", "--seed", default=42, type=int,
                    help="Random seed for corpus train/dev split (Default: 42)")
    ap = ap.parse_args()
    main(ap)
