import argparse
import os

from src.datasets.general_tsv import generate_corpus_files
from src.datasets.genia import get_cleaned_data_with_pos_dep
from src.utils.corpus_utils import create_vocab_from_embeddings


def tag_to_text_conll(tag):
    if tag == "PER":
        return "person"
    if tag == "ORG":
        return "organization"
    if tag == "LOC":
        return "location"
    return "miscellaneous"  # tag == "MISC"


def main(args):
    corpus_train = get_cleaned_data_with_pos_dep(os.path.join(args.dir, args.inp_train), delimiter=" ")
    corpus_dev = get_cleaned_data_with_pos_dep(os.path.join(args.dir, args.inp_dev), delimiter=" ")
    corpus_test = get_cleaned_data_with_pos_dep(os.path.join(args.dir, args.inp_test), delimiter=" ")
    corpus = corpus_train + corpus_dev + corpus_test

    generate_corpus_files(args.dir, corpus, corpus_train, corpus_dev, corpus_test, tag_to_text_fn=tag_to_text_conll)
    create_vocab_from_embeddings(embpath=args.glove_emb_path, vocabpath=os.path.join(args.dir, "glove_vocab.txt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CoNLL2003 corpus parser (convert to TSV)")
    ap.add_argument("--dir", type=str, default="../../data/CoNLL2003",
                    help="corpus dir (Default: '../../data/CoNLL2003')")
    ap.add_argument("--inp_train", type=str, default="train.bertner.txt",
                    help="path to input train (.txt) file, relative to corpus dir (Default: 'train.bertner.txt')")
    ap.add_argument("--inp_dev", type=str, default="dev.bertner.txt",
                    help="path to input dev (.txt) file, relative to corpus dir (Default: 'dev.bertner.txt')")
    ap.add_argument("--inp_test", type=str, default="test.bertner.txt",
                    help="path to input test (.txt) file, relative to corpus dir (Default: 'test.bertner.txt')")
    ap.add_argument("--glove_emb_path", type=str, default="../../../../Embeddings/glove.6B.50d.txt",
                    help="glove embeddings file path (Default: '../../../../Embeddings/glove.6B.50d.txt')")
    ap = ap.parse_args()
    main(ap)
