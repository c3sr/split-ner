import argparse
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from numpy.random import default_rng
from pyclustering.cluster.kmedoids import kmedoids
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import HfArgumentParser, AutoModel

from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataset
from secner.utils.general import setup_logging, parse_config, set_all_seeds, PairSpan

rng = default_rng(seed=42)


def embed(args, dataset, tag):
    model = AutoModel.from_pretrained(args.base_model).to("cuda:0")
    mention_dict = defaultdict(list)
    for index, sent in enumerate(dataset.sentences):
        if index % 100 == 0:
            print("processing sent: {0}".format(index))
        spans = get_spans(sent)
        if tag not in spans:
            continue
        inputs = torch.tensor([tok.bert_id for tok in sent.bert_tokens], device="cuda:0", dtype=torch.int64)
        out = model(inputs.unsqueeze(0))[0].squeeze(0)
        for sp in spans[tag]:
            m = dataset.tokenizer.decode(inputs[sp.start: sp.end + 1])
            v = torch.cat([out[sp.start, :], out[sp.end, :]]).detach().cpu().numpy()
            mention_dict[m].append(v)
    mentions = dict()
    for m in mention_dict:
        mentions[m] = np.mean(np.vstack(mention_dict[m]), axis=0)
    return mentions


def remap(args, dataset, tag, corpus_type, vecs):
    model = AutoModel.from_pretrained(args.base_model).to("cuda:0")
    for index, sent in enumerate(dataset.sentences):
        if index % 100 == 0:
            print("processing sent: {0}".format(index))
        spans = get_spans(sent)
        if tag not in spans:
            continue
        inputs = torch.tensor([tok.bert_id for tok in sent.bert_tokens], device="cuda:0", dtype=torch.int64)
        out = model(inputs.unsqueeze(0))[0].squeeze(0)
        for sp in spans[tag]:
            v = torch.cat([out[sp.start, :], out[sp.end, :]]).detach().cpu().numpy()
            k = np.argmin(np.array([distance.euclidean(vec, v) for vec in vecs]))
            for i in range(sp.start, sp.end + 1):
                curr = sent.tokens[sent.bert_tokens[i].token.offset]
                curr.tag = curr.tag[:2] + tag + str(k)

    with open("../../data/bio_cluster/{0}.tsv".format(corpus_type), "w") as f:
        for sent in dataset.sentences:
            for tok in sent.tokens:
                f.write("{0}\t{1}\t{2}\t{3}\n".format(tok.text, tok.pos_tag, tok.dep_tag, tok.tag))
            f.write("\n")


def tsne_plot(args, mentions, tag, corpus_type):
    vec = []
    labels = []
    for m in mentions:
        vec.append(mentions[m])
        labels.append(m)
    vec = np.vstack(vec)

    print("doing pca")
    pca_output = PCA(n_components=100).fit_transform(vec)
    print("doing tsne")
    tsne_output = TSNE(n_components=2).fit_transform(pca_output)
    plot_data = {"pca": pca_output, "x": tsne_output[:, 0], "y": tsne_output[:, 1], "group": labels, "vec": vec}

    os.makedirs("../../out/cluster", exist_ok=True)
    with open("../../out/cluster/{0}_{1}_{2}".format(args.dataset_dir, tag, corpus_type), "wb") as handle:
        pickle.dump(plot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved")

    plot_scatter(plot_data, args, tag, corpus_type)


def plot_scatter(plot_data, args, tag, corpus_type):
    plt.figure()
    ax = plt.subplot()
    sns.scatterplot(x="x", y="y",
                    data=plot_data,
                    color="skyblue",
                    ax=ax)
    for i in range(plot_data["x"].shape[0]):
        plt.text(plot_data["x"][i], plot_data["y"][i], plot_data["group"][i], fontsize=9, alpha=0.8)
    plt.savefig("../../out/cluster/{0}_{1}_{2}.png".format(args.dataset_dir, tag, corpus_type), dpi=500)
    plt.show()


def get_spans(sentence):
    spans = defaultdict(list)
    for index, tok in enumerate(sentence.bert_tokens):
        if tok.token.tag.startswith("B-"):
            spans[tok.token.tag[2:]].append(PairSpan(index, index))
        elif tok.token.tag.startswith("I-"):
            spans[tok.token.tag[2:]][-1].end = index
    return spans


def cluster(args, tag, corpus_type, approach):
    with open("../../out/cluster/{0}_{1}_{2}".format(args.dataset_dir, tag, corpus_type), "rb") as handle:
        plot_data = pickle.load(handle)

    K = len(plot_data["group"])

    print("tag: ", tag)
    print("approach: ", approach)

    tsne_data = np.hstack([np.array(plot_data["x"]).reshape(-1, 1), np.array(plot_data["y"]).reshape(-1, 1)])

    dp = [[0.0 for _ in range(K)] for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):

            if approach == "euclid-pca":
                dist = distance.euclidean(plot_data["pca"][i], plot_data["pca"][j])
            elif approach == "euclid-vec":
                dist = distance.euclidean(plot_data["vec"][i], plot_data["vec"][j])
            elif approach == "euclid-tsne":
                dist = distance.euclidean(tsne_data[i], tsne_data[j])
            elif approach == "cosine-pca":
                dist = distance.cosine(plot_data["pca"][i], plot_data["pca"][j])
            elif approach == "cosine-vec":
                dist = distance.cosine(plot_data["vec"][i], plot_data["vec"][j])
            elif approach == "cosine-tsne":
                dist = distance.cosine(tsne_data[i], tsne_data[j])
            else:
                continue

            dp[i][j] = dp[j][i] = dist
    print("created dist matrix")

    cluster_count = 4
    print("num clusters: ", cluster_count)
    inits = rng.choice(K, size=cluster_count, replace=False)
    # print("cluster inits:", inits)
    print("max iterations: ", 10)
    km_instance = kmedoids(dp, inits, data_type="distance_matrix", itermax=10)
    km_instance.process()
    clusters = km_instance.get_clusters()
    for clust in clusters:
        for item in clust:
            print(plot_data["group"][item], end=", ")
        print("\n")
    med = km_instance.get_medoids()
    print("medoids: ", med)
    med_vecs = [plot_data["vec"][m] for m in med]
    return med_vecs


def main(args):
    setup_logging()
    parser = HfArgumentParser([AdditionalArguments])
    additional_args = parse_config(parser, args.config)[0]
    additional_args.debug_mode = False
    set_all_seeds(42)
    corpus_type = "train"
    tag = "Gene_or_gene_product"

    dataset = NerDataset(additional_args, corpus_type=corpus_type)
    mentions = embed(additional_args, dataset, tag)
    tsne_plot(additional_args, mentions, tag, corpus_type)

    v = cluster(additional_args, tag, corpus_type, "euclid-vec")
    remap(additional_args, NerDataset(additional_args, "train"), tag, "train", v)
    remap(additional_args, NerDataset(additional_args, "dev"), tag, "dev", v)
    remap(additional_args, NerDataset(additional_args, "test"), tag, "test", v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mention Embedding Runner")
    ap.add_argument("--config", default="../config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
