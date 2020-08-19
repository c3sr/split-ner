from gensim.models import Word2Vec

from src.datasets.genia import *
from src.utils.corpus_utils import filter_embfile_using_tags_vocab, create_vocab_from_embeddings
from src.utils.tag_lm_embeddings import read_tags, generate_use_embeddings, write_emb_to_file, \
    generate_bert_embeddings, concatenate_w2v_with_use
from src.utils.word2vec import read_text_corpus


def create_mask_file_for_pipeline(mask_file_path):
    with open(mask_file_path, "w") as f:
        f.write("O\n")


def generate_corpus_files(root_dir, corpus, corpus_train, corpus_dev, corpus_test, tag_to_text_fn):
    write_to_file(corpus, os.path.join(root_dir, "corpus.tsv"))
    write_to_file(corpus_train, os.path.join(root_dir, "train.tsv"))
    write_to_file(corpus_dev, os.path.join(root_dir, "dev.tsv"))
    write_to_file(corpus_test, os.path.join(root_dir, "test.tsv"))

    generate_tags_vocab(corpus, os.path.join(root_dir, "tag_vocab.txt"))
    generate_all_tags_vocab(corpus, os.path.join(root_dir, "all_tag_vocab.txt"))
    generate_pos_tags_vocab(corpus, os.path.join(root_dir, "pos_tag_vocab.txt"))
    generate_dep_tags_vocab(corpus, os.path.join(root_dir, "dep_tag_vocab.txt"))

    emb_text_corpus_path = os.path.join(root_dir, "emb_text_corpus.txt")
    emb_vocab_path = os.path.join(root_dir, "emb_vocab.txt")
    generate_text_corpus_for_tag_embeddings(corpus_train, emb_text_corpus_path, emb_vocab_path)

    tag_dict = read_tags(os.path.join(root_dir, "all_tag_vocab.txt"), tag_to_text_fn=tag_to_text_fn)

    use_emb_dict = generate_use_embeddings(tag_dict)
    write_emb_to_file(use_emb_dict, os.path.join(root_dir, "tag_use_emb.txt"))

    scibert_emb_dict = generate_bert_embeddings(tag_texts=tag_dict, bert_model_name="allenai/scibert_scivocab_uncased")
    write_emb_to_file(scibert_emb_dict, os.path.join(root_dir, "tag_scibert_emb.txt"))

    bert_emb_dict = generate_bert_embeddings(tag_texts=tag_dict, bert_model_name="bert-base-uncased")
    write_emb_to_file(bert_emb_dict, os.path.join(root_dir, "tag_bert_emb.txt"))

    concatenate_w2v_with_use(use_emb_file=os.path.join(root_dir, "tag_use_emb.txt"),
                             w2v_emb_file=os.path.join(root_dir, "tag_w2v_emb.txt"),
                             out_file=os.path.join(root_dir, "tag_full_emb.txt"))

    sentences = read_text_corpus(emb_text_corpus_path)
    model = Word2Vec(sentences=sentences, size=50, window=10, min_count=1, workers=20, sample=1e-3, sg=1, hs=1,
                     negative=0, iter=10)
    model.wv.save_word2vec_format(os.path.join(root_dir, "emb_out.txt"), binary=False)
    filter_embfile_using_tags_vocab(os.path.join(root_dir, "emb_out.txt"), os.path.join(root_dir, "tag_w2v_emb.txt"),
                                    os.path.join(root_dir, "tag_vocab.txt"))

    open(os.path.join(root_dir, "empty_inp_tag_vocab.txt"), "w").close()
    open(os.path.join(root_dir, "empty_mask_tag_vocab.txt"), "w").close()

    create_mask_file_for_pipeline(os.path.join(root_dir, "mask_freq_tag_vocab.txt"))


def main(args):
    root_dir = os.path.join(args.parent_dir, args.corpus_name)
    corpus_train = get_cleaned_data_with_pos_dep(os.path.join(root_dir, args.train_path), delimiter=args.delimiter)
    corpus_dev = get_cleaned_data_with_pos_dep(os.path.join(root_dir, args.dev_path), delimiter=args.delimiter)
    corpus_test = get_cleaned_data_with_pos_dep(os.path.join(root_dir, args.test_path), delimiter=args.delimiter)
    corpus = corpus_train + corpus_dev + corpus_test

    generate_corpus_files(root_dir, corpus, corpus_train, corpus_dev, corpus_test,
                          tag_to_text_fn=lambda tag: " ".join(tag.lower().split("_")))
    create_vocab_from_embeddings(embpath=args.glove_emb_path, vocabpath=os.path.join(root_dir, "glove_vocab.txt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="General TSV input corpus converted into model format.")
    ap.add_argument("--parent_dir", default="../../../", type=str,
                    help="parent location of corpus dir (Default: '../../../')")
    ap.add_argument("--corpus_name", default="BioNLP13CG-IOB", type=str,
                    help="name of corpus directory, relative to parent location (Default: 'BioNLP13CG-IOB')")
    ap.add_argument("--train_path", default="orig_train.tsv", type=str,
                    help="train corpus input location, relative to corpus root (Default: 'orig_train.tsv')")
    ap.add_argument("--dev_path", default="orig_devel.tsv", type=str,
                    help="dev corpus input location, relative to corpus root (Default: 'orig_devel.tsv')")
    ap.add_argument("--test_path", default="orig_test.tsv", type=str,
                    help="test corpus input location, relative to corpus root (Default: 'orig_test.tsv')")
    ap.add_argument("--glove_emb_path", type=str, default="../../../../Embeddings/glove.6B.50d.txt",
                    help="glove embeddings file path (Default: '../../../../Embeddings/glove.6B.50d.txt')")
    ap.add_argument("--delimiter", type=str, default="\t",
                    help="delimiter in original input TSV file (\t|' ') (Default: '\t')")
    ap = ap.parse_args()
    main(ap)
