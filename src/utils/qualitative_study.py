import argparse
import csv
import os
from collections import defaultdict


class Entry:

    def __init__(self, token, gold, predicted):
        self.token = token
        self.gold = gold
        self.predicted = predicted


def read_data(filepath):
    data = []
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for index, row in enumerate(reader):
            if index == 0 or not row:
                continue
            data.append(Entry(token=row[0], gold=row[1], predicted=row[2]))
    return data


def analyse_misclassifications(data):
    mis = defaultdict(int)
    for d in data:
        if d.gold != d.predicted:
            mis[(d.gold, d.predicted)] += 1

    print("Gold\tPredicted\tCount")
    for key, value in sorted(mis.items(), key=lambda x: x[1], reverse=True):
        print("{0}\t{1}\t{2}".format(key[0], key[1], value))

    return mis


def get_gold_tag_diversity_for_token(data):
    div = defaultdict(set)
    for d in data:
        div[d.token].add(d.gold)

    cnt = 0
    for key, value in sorted(div.items(), key=lambda x: len(x[1]), reverse=True):
        # print("{0}\t{1}\t{2}".format(key, len(value), "\t".join(list(value))))
        cnt += 1
        if cnt > 300:
            break

    return div


def get_predicted_tag_diversity_for_token(data):
    div = defaultdict(set)
    for d in data:
        div[d.token].add(d.predicted)

    cnt = 0
    for key, value in sorted(div.items(), key=lambda x: len(x[1]), reverse=True):
        print("{0}\t{1}\t{2}".format(key, len(value), "\t".join(list(value))))
        cnt += 1
        if cnt > 300:
            break

    return div


def get_gold_token_diversity_for_tag(data):
    unique_cnt = defaultdict(set)
    total_cnt = defaultdict(int)

    for d in data:
        unique_cnt[d.gold].add(d.token)
        total_cnt[d.gold] += 1

    print("Tag\tUnique\tTotal")
    for key, value in sorted(unique_cnt.items(), key=lambda x: len(x[1]), reverse=True):
        print("{0}\t{1}\t{2}".format(key, len(value), total_cnt[key]))

    return unique_cnt, total_cnt


def get_all_unique_tokens_for_gold_tag(data, tag):
    unique_tokens = set()

    for d in data:
        if d.gold == tag:
            unique_tokens.add(d.token)

    for token in sorted(list(unique_tokens)):
        print(token)

    return list(unique_tokens)


def get_all_unique_misclassifications_for_tag(data, tag):
    unique_mis_classified_tokens = set()

    for d in data:
        if d.gold == tag and d.predicted != tag:
            unique_mis_classified_tokens.add(d.token)

    # for token in sorted(list(unique_mis_classified_tokens)):
    #     print(token)

    return list(unique_mis_classified_tokens)


def get_samples_for_tag_pair(data, gold_tag, predicted_tag):
    samples = defaultdict(int)

    for d in data:
        if d.gold == gold_tag and d.predicted == predicted_tag:
            samples[d.token] += 1

    print("Samples for Gold: {0} | Predicted: {1}".format(gold_tag, predicted_tag))
    print("Token\tCount")
    for token, cnt in sorted(samples.items(), key=lambda x: x[1], reverse=True):
        print("{0}\t{1}".format(token, cnt))

    return samples


def main(args):
    args.file = os.path.join(args.dir, args.file)
    data = read_data(args.file)
    analyse_misclassifications(data)

    get_samples_for_tag_pair(data, "B-G#other_name", "O")
    get_samples_for_tag_pair(data, "O", "I-G#DNA_domain_or_region")

    tok_div = get_gold_tag_diversity_for_token(data)
    get_predicted_tag_diversity_for_token(data)
    get_gold_token_diversity_for_tag(data)
    get_all_unique_tokens_for_gold_tag(data, "B-G#DNA_domain_or_region")

    mis = get_all_unique_misclassifications_for_tag(data, "I-ProteinMutation")

    for token in mis:
        if len(tok_div[token]) == 2:
            print("{0},{1}".format(token, len(tok_div[token])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Qualitative analysis utils")
    ap.add_argument("--dir", type=str,
                    default="../../../GENIA_term_3.02/quality/patcnn-word-pat-char",
                    help="Parent results directory (../../../tmVarCorpus/quality/tmvar-patcnn-word-pat-char"
                         "|../../../GENIA_term_3.02/quality/patcnn-word-pat-char)"
                         "(Default: '../../../GENIA_term_3.02/quality/patcnn-word-pat-char')")
    ap.add_argument("--file", type=str, default="dev.out.tsv",
                    help="Outputs filename relative to directory which needs to be analysed (Default: 'train.out.tsv')")
    ap = ap.parse_args()
    main(ap)
