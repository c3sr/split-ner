import argparse
import os
from collections import defaultdict

from sklearn.metrics import confusion_matrix


def parse_file(file_path):
    data = []
    with open(file_path, "r") as f:
        sent = []
        for line in f:
            line = line.strip()
            if not line:
                data.append(sent)
                sent = []
            else:
                s = line.split("\t")
                # token, gold, predicted
                sent.append([s[0], s[1], s[2]])
        return data


def calc_micro_f1(data):
    tp = defaultdict(list)
    fp = defaultdict(list)
    fn = defaultdict(list)
    gold_spans = [get_spans([[tok[0], tok[1]] for tok in sent]) for sent in data]
    pred_spans = [get_spans([[tok[0], tok[2]] for tok in sent]) for sent in data]
    print()
    for i in range(len(data)):
        for span in gold_spans[i]:
            if span in pred_spans[i]:
                tp[span[0]].append(span)
            else:
                fn[span[0]].append(span)
        for span in pred_spans[i]:
            if span not in gold_spans[i]:
                fp[span[0]].append(span)
    total_tp = sum([len(v) for v in tp.values()])
    total_fp = sum([len(v) for v in fp.values()])
    total_fn = sum([len(v) for v in fn.values()])
    p = total_tp * 1.0 / (total_tp + total_fp + 1e-7)
    r = total_tp * 1.0 / (total_tp + total_fn + 1e-7)
    f1 = 2.0 * p * r / (p + r + 1e-7)
    print("f1: {0:.4f}".format(100.0 * f1))
    return tp, fp, fn


def get_spans(tokens):
    spans = []
    for index, tok in enumerate(tokens):
        if tok[1].startswith("B-"):
            spans.append([tok[1][2:], [tok[0]], [index, index]])
            # spans.append([tok[1][2:], [], [index, index]]) # using this leads to some more matches.. may enquire cause
        elif tok[1].startswith("I-") and len(spans) > 0 and spans[-1][0] == tok[1][2:]:
            spans[-1][1].append(tok[0])
            spans[-1][2][1] = index
    return spans


def print_confusion_matrix_csv(data):
    labels = sorted(list(set([tok[1] for sent in data for tok in sent])))
    y_true = [tok[1] for sent in data for tok in sent]
    y_pred = [tok[2] for sent in data for tok in sent]
    mat = confusion_matrix(y_true, y_pred, labels)
    print(",".join(["#"] + labels))
    for i in range(len(labels)):
        print(",".join([labels[i]] + [str(k) for k in mat[i].tolist()]))


def analyse(data):
    print_confusion_matrix_csv(data)
    tp, fp, fn = calc_micro_f1(data)

    print_samples(tp, fp, fn, "Gene_or_gene_product")
    print_samples(tp, fp, fn, "Simple_chemical")
    print_samples(tp, fp, fn, "Cell")


def print_samples(tp, fp, fn, label):
    result = sorted([str(sp[1:]) for sp in fn[label]])
    print("False -ve ({0}) count: {1}".format(label, len(result)))
    print("\n".join(result))

    result = sorted([str(sp[1:]) for sp in fp[label]])
    print("False +ve ({0}) count: {1}".format(label, len(result)))
    print("\n".join(result))


def main(args):
    train_path = os.path.join(args.path, "train.tsv")
    dev_path = os.path.join(args.path, "dev1.tsv")  # "dev1" or "dev2" based on the mapping scheme defined in main.py
    test_path = os.path.join(args.path, "test.tsv")

    dev_data = parse_file(dev_path)
    analyse(dev_data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Predictions Analyzer")
    ap.add_argument("--path", type=str, default="../out/bio/ner-scibert/predictions",
                    help="dir with prediction outputs")
    ap = ap.parse_args()
    main(ap)
