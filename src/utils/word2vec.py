import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from scipy import spatial
from sklearn.manifold import TSNE

from src.utils.general import parse_emb_file


def read_text_corpus(corpus_path):
    sentences = []
    with open(corpus_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line.split())
    return sentences


def tsne(emb_path):
    emb_dict = parse_emb_file(emb_path, has_header_line=True)
    vocab = []
    emb = []
    for word, vec in emb_dict.items():
        vocab.append(word)
        emb.append(vec)
    emb = np.array(emb)

    print("Computing TSNE")
    tsne_vecs = TSNE(n_components=2).fit_transform(emb)

    print("Plotting TSNE")
    x = tsne_vecs[:, 0]
    y = tsne_vecs[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i in range(tsne_vecs.shape[0]):
        ax.annotate(vocab[i], (x[i], y[i]))
    plt.show()


def get_tag_similarity_distribution(emb_path, tag_vocab_path, has_header_line=True):
    emb_dict = parse_emb_file(emb_path, has_header_line=has_header_line)
    inf_cosine_dist = 2.0  # for tags which do not have an embedding

    tag_vocab = set()
    with open(tag_vocab_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("B-") or line.startswith("I-"):
                tag_vocab.add(line[2:])
            else:
                tag_vocab.add(line)
    tag_vocab = sorted(list(tag_vocab))

    for tag in tag_vocab:
        if tag not in emb_dict:
            continue
        dist = list()
        for other_tag in tag_vocab:
            if other_tag not in emb_dict:
                dist.append(inf_cosine_dist)
            else:
                dist.append(spatial.distance.cosine(emb_dict[tag], emb_dict[other_tag]))
        sorted_dist = sorted(enumerate(dist), key=lambda x: x[1])
        entries = ["{0} ({1:.3f})".format(tag_vocab[i], d) for i, d in sorted_dist[1:4]]
        print("{0},{1}".format(tag, ",".join(entries)))


def main(args):
    sentences = read_text_corpus(os.path.join(args.dir, args.corpus_path))

    print("Training Word2Vec")

    model = Word2Vec(sentences=sentences, size=args.dim, window=args.window, min_count=args.min_cnt, workers=20,
                     sample=1e-3, sg=1, hs=1, negative=0, iter=args.iter)

    print("Embeddings Trained")
    model.wv.save_word2vec_format(os.path.join(args.dir, args.out_emb_path), binary=False)

    tsne(os.path.join(args.dir, args.out_emb_path))

    get_tag_similarity_distribution(os.path.join(args.dir, args.out_emb_path),
                                    os.path.join(args.dir, args.tag_vocab_path), has_header_line=args.has_header_line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Word2Vec model")
    ap.add_argument("--dir", type=str, default="../../../GENIA_term_3.02",
                    help="root directory (Default: '../../../GENIA_term_3.02')")
    ap.add_argument("--corpus_path", type=str, default="emb_text_corpus.txt",
                    help="corpus text file path, relative to root directory "
                         "(emb_text_corpus.txt|std_emb_text_corpus.txt|jnlpba_emb_text_corpus.txt)"
                         "(Default: 'emb_text_corpus.txt')")
    ap.add_argument("--out_emb_path", type=str, default="emb_out.txt",
                    help="output text file with embeddings, relative to root directory "
                         "(emb_out.txt|std_emb_out.txt|jnlpba_emb_out.txt)"
                         "(Default: 'emb_out.txt')")
    ap.add_argument("--tag_vocab_path", type=str, default="tag_vocab.txt",
                    help="tag vocabulary file path, relative to root directory "
                         "(tag_vocab.txt|std_tag_vocab.txt|jnlpba_tag_vocab.txt)"
                         "(Default: 'tag_vocab.txt')")
    ap.add_argument("--dim", type=int, default=50, help="embedding dimensions (Default: 50)")
    ap.add_argument("--window", type=int, default=10, help="context window size (Default: 10)")
    ap.add_argument("--iter", type=int, default=10, help="no. of iterations over the corpus (Default: 10)")
    ap.add_argument("--has_header_line", action="store_true",
                    help="output embedding file has header line which is to be ignored during parsing (Default: False)")
    ap.add_argument("--min_cnt", type=int, default=1,
                    help="min count of a word for preparing its embeddings (Default: 1)")
    ap = ap.parse_args()
    main(ap)
