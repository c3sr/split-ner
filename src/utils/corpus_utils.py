import argparse
import csv
import os

import spacy
from spacy.tokens.doc import Doc

from src.utils.general import parse_emb_file


def create_vocab_from_embeddings(embpath, vocabpath):
    print("creating vocab file: {0} from supplied embeddings file: {1}".format(vocabpath, embpath))

    of = open(vocabpath, "w")
    with open(embpath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                word = line.split(" ")[0]
                of.write("{0}\n".format(word))
    of.close()


def get_token_tag_distribution(corpuspath, tags):
    tag_vocab = {t: 0 for t in tags}
    token_vocab = dict()

    if not corpuspath:
        return token_vocab, tag_vocab

    with open(corpuspath, "r") as f:
        for row in f:
            row = row.strip().split("\t")
            if len(row) == 0:
                continue
            token = row[0].lower()
            tag = row[-1]
            if token in token_vocab:
                token_vocab[token] += 1
            else:
                token_vocab[token] = 1

            if tag in tag_vocab:
                tag_vocab[tag] += 1
            else:
                tag_vocab[tag] = 1

    return token_vocab, tag_vocab


def calc_avg_word_seq_length(corpuspath):
    max_word_len = 0
    sum_word_len = 0
    word_cnt = 0

    max_sentence_len = 0
    sum_sentence_len = 0
    sentence_cnt = 0

    with open(corpuspath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        curr_sentence_len = 0
        for row in reader:
            if len(row) == 0:
                if curr_sentence_len > 0:
                    sum_sentence_len += curr_sentence_len
                    max_sentence_len = max(max_sentence_len, curr_sentence_len)
                    sentence_cnt += 1
                    curr_sentence_len = 0
            else:
                curr_sentence_len += 1
                max_word_len = max(max_word_len, len(row[0]))
                sum_word_len += len(row[0])
                word_cnt += 1

    avg_word_len = sum_word_len / word_cnt
    avg_sentence_len = sum_sentence_len / sentence_cnt

    print("Avg. Sentence Length: {0:.3f}".format(avg_sentence_len))
    print("Max Sentence Length: {0}".format(max_sentence_len))
    print("Avg. Word Length: {0:.3f}".format(avg_word_len))
    print("Max Word Length: {0}".format(max_word_len))


def get_dataset_stats(corpuspath, trainpath, devpath, testpath, tagspath):
    tags = []
    with open(tagspath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tags.append(line)

    corpus_token_vocab, corpus_tag_vocab = get_token_tag_distribution(corpuspath, tags)
    train_token_vocab, train_tag_vocab = get_token_tag_distribution(trainpath, tags)
    dev_token_vocab, dev_tag_vocab = get_token_tag_distribution(devpath, tags)
    test_token_vocab, test_tag_vocab = get_token_tag_distribution(testpath, tags)

    print("{0},{1},{2},{3},{4}".format("Property/Tag", "Corpus", "Train", "Dev", "Test"))
    print("{0},{1},{2},{3},{4}".format("Unique Tokens", len(corpus_token_vocab), len(train_token_vocab),
                                       len(dev_token_vocab), len(test_token_vocab)))
    for tag in corpus_tag_vocab:
        print("{0},{1},{2},{3},{4}".format(tag, corpus_tag_vocab[tag], train_tag_vocab[tag], dev_tag_vocab[tag],
                                           test_tag_vocab[tag]))

    print("\n")
    print("{0},{1}".format("Tag", "Count"))
    for tag in corpus_tag_vocab:
        if tag.startswith("B-"):
            print("{0},{1}".format(tag[2:], corpus_tag_vocab[tag], train_tag_vocab[tag], dev_tag_vocab[tag],
                                   test_tag_vocab[tag]))

    calc_avg_word_seq_length(corpuspath)

    print("corpus path: {0}".format(corpuspath))
    print("train path: {0}".format(trainpath))
    print("dev path: {0}".format(devpath))
    print("test path: {0}".format(testpath))


def filter_embfile_using_tags_vocab(emb_inpfile, emb_outfile, tagsfile):
    emb_dict = parse_emb_file(emb_inpfile)

    vocab = set()
    with open(tagsfile, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                if line.startswith("B-") or line.startswith("I-"):
                    vocab.add(line[2:])
                else:
                    vocab.add(line)

    with open(emb_outfile, "w") as f:
        for word in vocab:
            if word in emb_dict:
                f.write("{0} {1}\n".format(word, " ".join([str(x) for x in emb_dict[word]])))


def add_linguistic_features(corpus, dep=False, pos=False, spacy_model_name="en_core_sci_sm"):
    if not (dep or pos):
        print("no linguistic features to add")
        return corpus
    tokenizer_map = dict()
    nlp = spacy.load(spacy_model_name)
    nlp.tokenizer = lambda x: Doc(nlp.vocab, tokenizer_map[x])

    for i, sentence in enumerate(corpus):
        text_tokens = [token.text for token in sentence]
        sentence_text = " ".join(text_tokens)
        tokenizer_map[sentence_text] = text_tokens
        doc = nlp(sentence_text)
        for index in range(len(sentence)):
            if dep:
                sentence[index].dep_tag = doc[index].dep_
            if pos:
                sentence[index].pos_tag = doc[index].tag_
        if i % 1000 == 0:
            print("processed: {0} sentences".format(i))
    return corpus


def main(args):
    if args.task == 0:
        create_vocab_from_embeddings(args.emb_inpfile, args.vocabfile)
    elif args.task == 1:
        corpusfile = os.path.join(args.genia_path, args.corpusfile)
        trainfile = os.path.join(args.genia_path, args.trainfile)
        devfile = os.path.join(args.genia_path, args.devfile)
        testfile = os.path.join(args.genia_path, args.testfile)
        tagsfile = os.path.join(args.genia_path, args.tagsfile)
        get_dataset_stats(corpuspath=corpusfile, trainpath=trainfile, devpath=devfile, testpath=testfile,
                          tagspath=tagsfile)
    elif args.task == 2:
        corpusfile = os.path.join(args.tmvar_path, args.corpusfile)
        trainfile = os.path.join(args.tmvar_path, args.trainfile)
        testfile = os.path.join(args.tmvar_path, args.testfile)
        tagsfile = os.path.join(args.tmvar_path, args.tagsfile)
        get_dataset_stats(corpuspath=corpusfile, trainpath=trainfile, devpath=None, testpath=testfile,
                          tagspath=tagsfile)
    elif args.task == 3:
        tagsfile = os.path.join(args.genia_path, args.tagsfile)
        filter_embfile_using_tags_vocab(args.emb_inpfile, args.emb_outfile, tagsfile)


if __name__ == "__main__":
    tasks = ["0: create vocab file from words in supplied emb. file",
             "1: generate token/tag stats for GENIA dataset(input in TSV form)",
             "2: generate token/tag stats for TmVar dataset(input in TSV form)",
             "3: filter tag embeddings input file to retain only words in supplied tags vocab file"]
    ap = argparse.ArgumentParser(description="Corpus processing utils. Tasks supported:" + "\n".join(tasks))
    ap.add_argument("-t", "--task", type=int, default=1, help="Task ID (Default: 1)")
    ap.add_argument("--genia_path", type=str, default="../../../GENIA_term_3.02",
                    help="path to GENIA corpus dir (Default: '../../../GENIA_term_3.02')")
    ap.add_argument("--tmvar_path", type=str, default="../../../tmVarCorpus",
                    help="path to TmVar corpus dir (Default: '../../../tmVarCorpus')")
    ap.add_argument("--corpusfile", type=str, default="corpus.tsv",
                    help="relative path to corpus file (corpus.tsv|std_corpus.tsv|jnlpba_corpus.tsv) "
                         "(Default: 'corpus.tsv')")
    ap.add_argument("--trainfile", type=str, default="train.tsv",
                    help="relative path to train file (train.tsv|std_train.tsv|jnlpba_train.tsv) "
                         "(Default: 'train.tsv')")
    ap.add_argument("--devfile", type=str, default="dev.tsv",
                    help="relative path to dev file (dev.tsv|std_dev.tsv|jnlpba_dev.tsv) (Default: 'dev.tsv')")
    ap.add_argument("--testfile", type=str, default="test.tsv",
                    help="relative path to test file (test.tsv|std_test.tsv|jnlpba_test.tsv) (Default: 'test.tsv')")
    ap.add_argument("--tagsfile", type=str, default="tag_vocab.txt",
                    help="relative path to tags vocab file (tag_vocab.txt|std_tag_vocab.txt|jnlpba_tag_vocab.txt) "
                         "(Default: 'tag_vocab.txt')")
    ap.add_argument("--vocabfile", type=str, default="../../../GENIA_term_3.02/glove_vocab.txt",
                    help="output token vocab file path (Default: '../../../GENIA_term_3.02/glove_vocab.txt')")
    ap.add_argument("--emb_inpfile", type=str, default="../../../../Embeddings/glove.6B.50d.txt",
                    help="embeddings input file path "
                         "('../../../GENIA_term_3.02/emb_out.txt'|'../../../../Embeddings/glove.6B.50d.txt') "
                         "(Default: '../../../../Embeddings/glove.6B.50d.txt')")
    ap.add_argument("--emb_outfile", type=str, default="../../../GENIA_term_3.02/tag_w2v_emb.txt",
                    help="embeddings output file path (tag_w2v_emb.txt|std_tag_w2v_emb.txt|jnlpba_tag_w2v_emb.txt) "
                         "(Default: '../../../GENIA_term_3.02/tag_w2v_emb.txt')")
    ap = ap.parse_args()
    main(ap)
