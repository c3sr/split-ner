import argparse
import os

from src.datasets.general_tsv import generate_corpus_files
from src.datasets.genia import get_cleaned_data_with_pos_dep
from src.utils.corpus_utils import create_vocab_from_embeddings


def tag_to_text_onto(tag):
    if tag == "CARDINAL":
        return "cardinal"
    if tag == "DATE":
        return "date"
    if tag == "EVENT":
        return "event"
    if tag == "FAC":
        return "facility"
    if tag == "GPE":
        return "geo-political entity"
    if tag == "LANGUAGE":
        return "language"
    if tag == "LAW":
        return "law"
    if tag == "LOC":
        return "location"
    if tag == "MONEY":
        return "money"
    if tag == "NORP":
        return "nationalities or religious or political groups"
    if tag == "ORDINAL":
        return "ordinal"
    if tag == "ORG":
        return "organization"
    if tag == "PERCENT":
        return "percent"
    if tag == "PERSON":
        return "person"
    if tag == "PRODUCT":
        return "product"
    if tag == "QUANTITY":
        return "quantity"
    if tag == "TIME":
        return "time"
    return "work of art"  # tag == "WORK_OF_ART"


def main(args):
    corpus_train = get_cleaned_data_with_pos_dep(os.path.join(args.dir, args.inp_train), delimiter="\t")
    corpus_dev = get_cleaned_data_with_pos_dep(os.path.join(args.dir, args.inp_dev), delimiter="\t")
    corpus_test = get_cleaned_data_with_pos_dep(os.path.join(args.dir, args.inp_test), delimiter="\t")
    corpus = corpus_train + corpus_dev + corpus_test

    generate_corpus_files(args.dir, corpus, corpus_train, corpus_dev, corpus_test, tag_to_text_fn=tag_to_text_onto)
    create_vocab_from_embeddings(embpath=args.glove_emb_path, vocabpath=os.path.join(args.dir, "glove_vocab.txt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OntoNotes corpus parser (convert to TSV)")
    ap.add_argument("--dir", type=str, default="../../data/OntoNotes-5.0-NER-BIO",
                    help="corpus dir (Default: '../../data/OntoNotes-5.0-NER-BIO')")
    ap.add_argument("--inp_train", type=str, default="onto.train.ner",
                    help="path to input train (.txt) file, relative to corpus dir (Default: 'onto.train.ner')")
    ap.add_argument("--inp_dev", type=str, default="onto.development.ner",
                    help="path to input dev (.txt) file, relative to corpus dir (Default: 'onto.development.ner')")
    ap.add_argument("--inp_test", type=str, default="onto.test.ner",
                    help="path to input test (.txt) file, relative to corpus dir (Default: 'onto.test.ner')")
    ap.add_argument("--glove_emb_path", type=str, default="../../../../Embeddings/glove.6B.50d.txt",
                    help="glove embeddings file path (Default: '../../../../Embeddings/glove.6B.50d.txt')")
    ap = ap.parse_args()
    main(ap)
