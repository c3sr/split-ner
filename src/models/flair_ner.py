import argparse
import os

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import CharacterEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


def train(args):
    cols = {0: 'text', 1: 'ner'}
    corpus: Corpus = ColumnCorpus(args.datapath, cols, train_file=args.trainfile, test_file=args.testfile,
                                  dev_file=args.devfile, column_delimiter="\t|\n")
    print(corpus)

    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary)

    embedding_types = [
        # WordEmbeddings('glove'),
        CharacterEmbeddings(),
        # FlairEmbeddings('news-forward'),
        # FlairEmbeddings('news-backward'),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=args.hidden_dim,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    print("initiating training")
    trainer.train(args.checkpoint_dir,
                  learning_rate=args.lr,
                  mini_batch_size=args.batch_size,
                  max_epochs=args.num_epochs)
    print("done")


def query(args):
    if args.query == "best":
        model_path = os.path.join(args.checkpoint_dir, 'best-model.pt')
    else:
        model_path = os.path.join(args.checkpoint_dir, 'final-model.pt')

    model = SequenceTagger.load(model_path)
    sentence_text = input("Enter Query Sentence:")
    sentence = Sentence(sentence_text)
    model.predict(sentence)
    print("RESPONSE: {0}".format(sentence.to_tagged_string()))


def main(args):
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)
    if args.query == "none":
        train(args)
        return
    query(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CNN-BiLSTM-CRF model for NER with Flair Embeddings")
    ap.add_argument("--name", type=str, default="flair", help="model name (Default: 'flair')")
    ap.add_argument("--datapath", type=str, default="../../../GENIA_term_3.02",
                    help="corpus directory (Default: '../../../GENIA_term_3.02')")
    ap.add_argument("--checkpoint_dir", type=str, default="../../checkpoints",
                    help="checkpoint directory (Default: '../../checkpoints')")
    ap.add_argument("--trainfile", type=str, default="train.tsv",
                    help="train file within corpus directory (Default: 'train.tsv')")
    ap.add_argument("--devfile", type=str, default="dev.tsv",
                    help="dev file within corpus directory (Default: 'dev.tsv')")
    ap.add_argument("--testfile", type=str, default="test.tsv",
                    help="test file within corpus directory (Default: 'test.tsv')")
    ap.add_argument("--query", type=str, default="none",
                    help="run qualitative analysis on existing model. Select 'none' for training (none|final|best) "
                         "(Default: 'none')")

    ap.add_argument("--batch_size", type=int, default=128, help="batch size (Default: 128)")
    ap.add_argument("--num_epochs", type=int, default=150, help="# epochs to train (Default: 150)")
    ap.add_argument("--lr", type=float, default=0.001, help="corpus directory (Default: 0.001)")
    ap.add_argument("--hidden_dim", type=int, default=256, help="hidden state dimension (Default: 256)")
    ap = ap.parse_args()
    main(ap)
