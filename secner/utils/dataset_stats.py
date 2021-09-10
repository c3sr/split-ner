import argparse
import os
import re
from collections import defaultdict

from secner.utils.general import read_data


def get_mention_density(data):
    cnt_mention = 0
    for sent in data:
        for tok in sent:
            if tok[-1].startswith("B-"):
                cnt_mention += 1

    print("cnt_mention: ", cnt_mention)
    print("len_data: ", len(data))
    
    return cnt_mention / len(data)


def main(args):
    data_dir = os.path.join("..", "..", "data", args.path)
    train_data = read_data(os.path.join(data_dir, "train.tsv"))
    dev_data = read_data(os.path.join(data_dir, "dev.tsv"))
    test_data = read_data(os.path.join(data_dir, "test.tsv"))

    print("# Sentences: {0} (Train) | {1} (Dev) | {2} (Test)".format(len(train_data), len(dev_data), len(test_data)))
    corpus_data = train_data + dev_data + test_data
    print("mention density: {0:.2f}".format(get_mention_density(corpus_data)))
    print("Avg. sentence length: {0:.3f}".format(sum(len(sent) for sent in corpus_data) * 1.0 / len(corpus_data)))

    all_tokens = [(tok[0], tok[-1]) for sent in corpus_data for tok in sent]
    all_mention_tokens = [tup for tup in all_tokens if tup[1] != "O"]
    all_none_tokens = [tup for tup in all_tokens if tup[1] == "O"]
    print("ratio of NON O-labeled tokens: {0:.4f}%".format(100.0 - (len(all_none_tokens) * 100.0 / len(all_tokens))))
    alpha_num_tokens = [tup for tup in all_mention_tokens if (tup[1] != "O" and bool(re.search(r"(\d)", tup[0])))]
    print("Alphanumeric entities: {0:.3f}%".format(len(alpha_num_tokens) * 100.0 / len(all_mention_tokens)))
    non_alpha_tokens = [tup for tup in all_mention_tokens if (tup[1] != "O" and bool(re.search(r"[^A-Za-z]", tup[0])))]
    print("Non-alphabetic(^A-Za-z) entities: {0:.3f}%".format(len(non_alpha_tokens) * 100.0 / len(all_mention_tokens)))

    tag_counts = defaultdict(int)
    for tup in all_mention_tokens:
        if tup[1].startswith("B-"):
            tag_counts[tup[1][2:]] += 1
    print("Tag distribution: {0}".format(tag_counts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dataset Stats")
    ap.add_argument("--path", type=str, default="bio", help="dataset folder")
    ap = ap.parse_args()
    main(ap)
