import argparse
import os
import re

from secner.utils.general import Token, Sentence


def read_text(filename):
    return "".join([line for line in open(filename, "r", encoding="utf-8")])


def read_annotations(filename):
    tags = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s = line.split("\t")
                t = s[1].split()
                tag, start, end, phrase = t[0], int(t[1]), int(t[2]), s[2]
                tags.append([tag, start, end, phrase])
    return tags


def make_entry(text, tags, doc_id):
    tokens = re.split("([,;.!?:'\"/|_@#$%^&*~`+-=<>()\[\]{}]|\s+)", text)
    token_with_pos = []
    end = 0
    for tok in tokens:
        start = end
        end = start + len(tok)
        tok = tok.strip()
        if tok:
            token_with_pos.append((Token(tok, "O"), (start, end)))
    error_cnt = 0
    non_overlap_tags = []
    start = 0
    for tag in tags:
        if tag[1] >= start:
            non_overlap_tags.append(tag)
            start = tag[2]
        else:
            print("error overlap: {0}, {1}, {2}, {3}".format(doc_id, tag, start, non_overlap_tags[-1]))
            error_cnt += 1

    for tag in non_overlap_tags:
        found_start, found_end = False, False
        for tok in token_with_pos:
            if tok[1][0] == tag[1]:
                tok[0].tag = "B-{0}".format(tag[0])
                found_start = True
            elif found_start and tok[1][0] > tag[1] and tok[1][1] <= tag[2]:
                tok[0].tag = "I-{0}".format(tag[0])
            if tok[1][1] == tag[2]:
                found_end = True
                break

        if not (found_start and found_end):
            print("error boundary: ", doc_id, tag)
            error_cnt += 1

    final_tokens = [tok[0] for tok in token_with_pos]

    sentences = []
    start = 0
    do_split = False
    for i in range(len(final_tokens)):
        if i - start >= 300:
            do_split = True
        if do_split and final_tokens[i].text == ".":
            sentences.append(Sentence(final_tokens[start: i + 1]))
            start = i + 1
            do_split = False
    if start < len(final_tokens):
        sentences.append(Sentence(final_tokens[start:]))

    return sentences, error_cnt


def read_data(path, corpus_type):
    dir_path = os.path.join(path, "raw", "BioNLP-ST_2013_CG_{0}_data".format(corpus_type))
    data = []
    all_tags_cnt = 0
    all_overlap_cnt = 0
    for f in os.listdir(dir_path):
        match = re.match(r"PMID-(\d+)\.txt", f)
        if match:
            doc_id = match.group(1)
            text = read_text(os.path.join(dir_path, f))
            tags = read_annotations(os.path.join(dir_path, "PMID-{0}.a1".format(doc_id)))
            all_tags_cnt += len(tags)
            sentences, overlap_cnt = make_entry(text, tags, doc_id)
            all_overlap_cnt += overlap_cnt
            data.extend(sentences)
    print("# annotations: {0}".format(all_tags_cnt))
    print("# overlap annotations: {0}".format(all_overlap_cnt))
    return data


def out_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for sent in data:
            for tok in sent.tokens:
                f.write("{0}\t{1}\n".format(tok.text, tok.tag))
            f.write("\n")


def main(args):
    train_data = read_data(args.path, "training")
    dev_data = read_data(args.path, "development")
    test_data = read_data(args.path, "test")

    out_data(train_data, os.path.join(args.path, "train.tsv"))
    out_data(dev_data, os.path.join(args.path, "dev.tsv"))
    out_data(test_data, os.path.join(args.path, "test.tsv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BioNLP13CG corpus parser")
    ap.add_argument("--path", type=str, default="../../data/bio_context", help="raw data path")
    ap = ap.parse_args()
    main(ap)
