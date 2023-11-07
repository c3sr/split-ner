import argparse
from collections import defaultdict

from splitner.additional_args import AdditionalArguments
from splitner.dataset import NerDataset
from splitner.utils.general import Sentence, Token


def mine_patterns(data_spans, tag):
    tag_spans = []
    for sent_spans in data_spans:
        for span in sent_spans:
            if span[0] == tag:
                tag_spans.append(" ".join(span[2]))
    sorted_tag_spans = sorted(tag_spans)
    print("\n".join(sorted_tag_spans))


def intrinsic_patterns(args):
    default_additional_args = AdditionalArguments()
    data = NerDataset.read_dataset("../../data/{0}/{1}.tsv".format(args.corpus, args.file), default_additional_args)
    term_dict = dict()
    for sent in data:
        for tok in sent.tokens:
            text = NerDataset.make_pattern_type0(tok.text)
            if text not in term_dict:
                term_dict[text] = defaultdict(int)
            tag = tok.tags[-1] if tok.tags[-1] == "O" else tok.tags[-1][2:]
            term_dict[text][tag] += 1
    term_dict_percent = dict()
    for term in term_dict:
        d = sum(term_dict[term].values())
        tmp = {key: round(val * 100.0 / d, 1) for key, val in term_dict[term].items() if val / d >= 0.1}
        if (len(tmp.keys()) == 1 and "O" in tmp) or d < 10:
            continue
        term_dict_percent[term] = (d, dict(sorted(tmp.items(), key=lambda item: -item[1])))
    term_dict_percent = dict(sorted(term_dict_percent.items(), key=lambda item: len(item[1][1])))
    print("Num Items: {0}".format(len(term_dict_percent)))
    for k in term_dict_percent:
        print("{0}\t{1}\t{2}".format(k, term_dict_percent[k][0], term_dict_percent[k][1]))


def extrinsic_patterns(args):
    default_additional_args = AdditionalArguments()
    data = NerDataset.read_dataset("../../data/{0}/{1}.tsv".format(args.corpus, args.file), default_additional_args)
    new_data = []
    for sent in data:
        new_tokens = []
        for i in range(len(sent.tokens)):
            if sent.tokens[i].tags[-1].startswith("B-"):
                new_tokens.append(Token(sent.tokens[i].tags[-1][2:], sent.tokens[i].tags[-1][2:]))
            elif sent.tokens[i].tags[-1] == "O":
                new_tokens.append(Token(sent.tokens[i].text, "O"))
        new_data.append(Sentence(new_tokens))

    pat_dict = defaultdict(int)
    for sent in new_data:
        for i in range(len(sent.tokens)):
            tag = sent.tokens[i].tags[-1]
            if tag == "O":
                continue
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 1), min(len(sent.tokens), i + 1))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 1), min(len(sent.tokens), i + 2))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 1), min(len(sent.tokens), i + 3))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 1), min(len(sent.tokens), i + 4))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 1), min(len(sent.tokens), i + 5))])] += 1

            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 2), min(len(sent.tokens), i + 1))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 2), min(len(sent.tokens), i + 2))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 2), min(len(sent.tokens), i + 3))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 2), min(len(sent.tokens), i + 4))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 2), min(len(sent.tokens), i + 5))])] += 1

            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 3), min(len(sent.tokens), i + 1))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 3), min(len(sent.tokens), i + 2))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 3), min(len(sent.tokens), i + 3))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 3), min(len(sent.tokens), i + 4))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 3), min(len(sent.tokens), i + 5))])] += 1

            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 4), min(len(sent.tokens), i + 1))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 4), min(len(sent.tokens), i + 2))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 4), min(len(sent.tokens), i + 3))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 4), min(len(sent.tokens), i + 4))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 4), min(len(sent.tokens), i + 5))])] += 1

            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 5), min(len(sent.tokens), i + 1))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 5), min(len(sent.tokens), i + 2))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 5), min(len(sent.tokens), i + 3))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 5), min(len(sent.tokens), i + 4))])] += 1
            pat_dict[" ".join([sent.tokens[j].text for j in range(max(0, i - 5), min(len(sent.tokens), i + 5))])] += 1

    pat_dict = dict(sorted(pat_dict.items(), key=lambda item: -len(item[0])))
    print("#All Entries: {0}".format(sum(pat_dict.values())))
    for pat in pat_dict:
        if pat_dict[pat] >= 40:
            print("{0}: {1}".format(pat_dict[pat], pat))
    # mine_patterns(spans, "Amino_acid")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Frequent pattern mining over corpus")
    ap.add_argument("--corpus", type=str, default="bio")
    ap.add_argument("--file", type=str, default="test")
    ap = ap.parse_args()
    # intrinsic_patterns(ap)
    extrinsic_patterns(ap)
