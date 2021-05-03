import argparse
import os
from collections import defaultdict

from sklearn.metrics import confusion_matrix


def parse_file(file_path):
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, "r", encoding="utf-8") as f:
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


def make_gold_file(path, outfile, data):
    with open(os.path.join(path, outfile), "w", encoding="utf-8") as f:
        for sent in data:
            for tok in sent:
                f.write("{0}\t{1}\t{2}\n".format(tok[0], tok[1], tok[1]))
            f.write("\n")


def calc_micro_f1(data):
    tp = defaultdict(list)
    fp = defaultdict(list)
    fn = defaultdict(list)
    gold_spans = [span_corrector(get_spans([[tok[0], tok[1]] for tok in sent], index)) for index, sent in
                  enumerate(data)]
    pred_spans = [span_corrector(get_spans([[tok[0], tok[2]] for tok in sent], index)) for index, sent in
                  enumerate(data)]
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
    total_cnt = total_tp + total_fn
    p = total_tp * 1.0 / (total_tp + total_fp + 1e-7)
    r = total_tp * 1.0 / (total_tp + total_fn + 1e-7)
    f1 = 2.0 * p * r / (p + r + 1e-7)
    print("Overall | Cnt: {0} | P: {1:.4f} | R: {2:.4f} | Micro F1: {3:.4f}"
          .format(total_cnt, 100.0 * p, 100.0 * r, 100.0 * f1))

    tags = list(set(tp.keys()).union(set(fn.keys())))
    tags = sorted(tags, key=lambda x: len(tp[x]) + len(fn[x]), reverse=True)
    tag_f1_sum = 0.0
    for tag in tags:
        tag_tp, tag_fp, tag_fn = len(tp[tag]), len(fp[tag]), len(fn[tag])
        tag_cnt = tag_tp + tag_fn
        tag_p = tag_tp * 1.0 / (tag_tp + tag_fp + 1e-7)
        tag_r = tag_tp * 1.0 / (tag_tp + tag_fn + 1e-7)
        tag_f1 = 2.0 * tag_p * tag_r / (tag_p + tag_r + 1e-7)
        tag_f1_sum += tag_f1
        print("Tag: {0} | Cnt: {1} | P: {2:.4f} | R: {3:.4f} | F1: {4:.4f}"
              .format(tag, tag_cnt, 100.0 * tag_p, 100.0 * tag_r, 100.0 * tag_f1))
    print("Macro F1: {0:.4f}".format(tag_f1_sum * 100.0 / len(tags)))

    return tp, fp, fn


def get_spans(tokens, sent_index):
    spans = []
    can_continue = False
    for index, tok in enumerate(tokens):
        if tok[1].startswith("B-"):
            spans.append([tok[1][2:], sent_index, [tok[0]], [index, index]])
            can_continue = True
            # using the below representation leads to more matches.. may enquire cause
            # spans.append([tok[1][2:], sent_index, [], [index, index]])
        elif tok[1].startswith("I-") and len(spans) > 0 and spans[-1][0] == tok[1][2:] and can_continue:
            spans[-1][2].append(tok[0])
            spans[-1][3][1] = index
        else:
            can_continue = False
    return spans


# can check special corrections and their effect on overall F1 through this
def span_corrector(spans):
    # for span in spans:
    #     if "FVIIa" in span[2]:
    #         span[0] = "Gene_or_gene_product"
    return spans


def print_confusion_matrix_csv(data):
    print("CONFUSION MATRIX:")
    labels = sorted(list(set([tok[1] for sent in data for tok in sent])))
    y_true = [tok[1] for sent in data for tok in sent]
    y_pred = [tok[2] for sent in data for tok in sent]
    mat = confusion_matrix(y_true, y_pred, labels)
    print(",".join(["#"] + labels))
    for i in range(len(labels)):
        print(",".join([labels[i]] + [str(k) for k in mat[i].tolist()]))


def print_error_samples(tp, fp, fn, tag):
    result = sorted([str(span[1:]) for span in fn[tag]])
    print("#False -ve ({0}): {1}".format(tag, len(result)))
    print("\n".join(result))
    print()
    result = sorted([str(span[1:]) for span in fp[tag]])
    print("#False +ve ({0}): {1}".format(tag, len(result)))
    print("\n".join(result))


def print_all_gold_samples(data, tag):
    st = set()
    entity = ""
    for sent in data:
        for tok in sent:
            if tok[1] == "B-{0}".format(tag):
                if entity:
                    st.add(entity)
                entity = tok[0]
            elif tok[1] == "I-{0}".format(tag):
                entity += " " + tok[0]
    if entity:
        st.add(entity)
    print("GOLD SAMPLES ({0}): {1}".format(tag, sorted(list(st))))


def analyse_errors(data):
    print_confusion_matrix_csv(data)
    tp, fp, fn = calc_micro_f1(data)

    # BioNLP13CG
    print_error_samples(tp, fp, fn, "Simple_chemical")
    # print_error_samples(tp, fp, fn, "Gene_or_gene_product")
    # print_error_samples(tp, fp, fn, "Cell")

    print_all_gold_samples(data, "Simple_chemical")

    # JNLPBA
    # print_error_samples(tp, fp, fn, "protein")


def analyse_overlap_errors_for_tag(data, tag, verbose=True, outfile=None):
    gold_spans = [get_spans([[tok[0], tok[1]] for tok in sent], index) for index, sent in enumerate(data)]
    pred_spans = [get_spans([[tok[0], tok[2]] for tok in sent], index) for index, sent in enumerate(data)]
    overlap_error_cases = []
    num_multi_word_gold_spans_with_overlap_errors = 0
    for gold_sent, pred_sent in zip(gold_spans, pred_spans):
        for g_span in gold_sent:
            if not g_span[0] == tag:
                continue
            is_multi_word_with_overlap_error = False
            is_fully_missed = True
            for p_span in pred_sent:
                if p_span == g_span:
                    is_fully_missed = False
                elif not (p_span[-1][0] > g_span[-1][1] or p_span[-1][1] < g_span[-1][0]):
                    is_fully_missed = False
                    overlap_error_cases.append((g_span, p_span))
                    if g_span[-1][0] < g_span[-1][1]:
                        is_multi_word_with_overlap_error = True
            if is_multi_word_with_overlap_error:
                num_multi_word_gold_spans_with_overlap_errors += 1
            if is_fully_missed:
                overlap_error_cases.append((g_span, None))

        for p_span in pred_sent:
            if not p_span[0] == tag:
                continue
            is_fully_noise = True
            for g_span in gold_sent:
                if not (p_span[-1][0] > g_span[-1][1] or p_span[-1][1] < g_span[-1][0]):
                    is_fully_noise = False
                    break
            if is_fully_noise:
                overlap_error_cases.append((None, p_span))

    if verbose:
        print("Tag: {0}".format(tag))
        print("#Overlap Errors: {0}".format(len(overlap_error_cases)))
        print("#Multi-word gold spans with overlap errors: {0}".format(num_multi_word_gold_spans_with_overlap_errors))
        # print_overlap_error_spans(overlap_error_cases, data)

    # Error Segregation
    fully_noise = []
    missed_completely = []

    missed_prefix_missed_suffix = []
    missed_prefix_only = []
    missed_prefix_extra_suffix = []

    missed_suffix_only = []
    exact_match_tag_diff = []
    extra_suffix_only = []

    extra_prefix_missed_suffix = []
    extra_prefix_only = []
    extra_prefix_extra_suffix = []

    for g_span, p_span in overlap_error_cases:
        if not g_span:
            fully_noise.append((g_span, p_span))
        elif not p_span:
            missed_completely.append((g_span, p_span))
        elif g_span[3][0] < p_span[3][0]:
            if g_span[3][1] > p_span[3][1]:
                missed_prefix_missed_suffix.append((g_span, p_span))
            elif g_span[3][1] == p_span[3][1]:
                missed_prefix_only.append((g_span, p_span))
            else:
                missed_prefix_extra_suffix.append((g_span, p_span))
        elif g_span[3][0] == p_span[3][0]:
            if g_span[3][1] > p_span[3][1]:
                missed_suffix_only.append((g_span, p_span))
            elif g_span[3][1] == p_span[3][1]:
                exact_match_tag_diff.append((g_span, p_span))
            else:
                extra_suffix_only.append((g_span, p_span))
        else:
            if g_span[3][1] > p_span[3][1]:
                extra_prefix_missed_suffix.append((g_span, p_span))
            elif g_span[3][1] == p_span[3][1]:
                extra_prefix_only.append((g_span, p_span))
            else:
                extra_prefix_extra_suffix.append((g_span, p_span))

    if verbose:
        if outfile and os.path.exists(outfile):
            os.remove(outfile)
        print_error_overlap_spans(missed_completely, data, "MISSED COMPLETELY", outfile)
        print_error_overlap_spans(missed_prefix_missed_suffix, data, "MISSED PREFIX, MISSED SUFFIX", outfile)
        print_error_overlap_spans(missed_prefix_only, data, "MISSED PREFIX", outfile)
        print_error_overlap_spans(missed_prefix_extra_suffix, data, "MISSED PREFIX, EXTRA SUFFIX", outfile)
        print_error_overlap_spans(missed_suffix_only, data, "MISSED SUFFIX", outfile)
        print_error_overlap_spans(exact_match_tag_diff, data, "EXACT MATCH, TAG DIFFERENT", outfile)
        print_error_overlap_spans(extra_suffix_only, data, "EXTRA SUFFIX", outfile)
        print_error_overlap_spans(extra_prefix_missed_suffix, data, "EXTRA PREFIX, MISSED SUFFIX", outfile)
        print_error_overlap_spans(extra_prefix_only, data, "EXTRA PREFIX", outfile)
        print_error_overlap_spans(extra_prefix_extra_suffix, data, "EXTRA PREFIX, EXTRA SUFFIX", outfile)
        print_error_overlap_spans(fully_noise, data, "FULLY NOISE", outfile)

    return num_multi_word_gold_spans_with_overlap_errors


def print_error_overlap_spans(error_spans, data, heading, outfile=None):
    if len(error_spans) == 0:
        return
    log_str = "{0}\n".format(heading)
    for g_span, p_span in error_spans:
        sent_index = g_span[1] if g_span else p_span[1]
        sent_text = " ".join([tok[0] for tok in data[sent_index]])
        g_span_str = "({0})'{1}'".format(g_span[0], " ".join(g_span[2])) if g_span else "-"
        p_span_str = "({0})'{1}'".format(p_span[0], " ".join(p_span[2])) if p_span else "-"
        log_str += "gold: {0}\tpred: {1}\tdata[{2}]: {3}\n".format(g_span_str, p_span_str, sent_index,
                                                                   sent_text).replace("\"", "'")
    log_str += "\n"
    if outfile:
        with open(outfile, "a", encoding="utf-8") as f:
            f.write(log_str)
    else:
        print(log_str, end="")


def print_overlap_error_stats(error_dict, data):
    total_error_cnt = 0
    num_overlap_error_gold_spans = 0
    for tag in error_dict:
        total_error_cnt += len(error_dict[tag])
        tag_overlap_error_gold_span_cnt = analyse_overlap_errors_for_tag(data, tag, verbose=False)
        num_overlap_error_gold_spans += tag_overlap_error_gold_span_cnt
        print("Tag: {0}, #Errors: {1}, #Overlap Errors: {2}"
              .format(tag, len(error_dict[tag]), tag_overlap_error_gold_span_cnt))

    # note: this ratio is not fully correct. Numerator is not fully contained in denominator!
    ratio = num_overlap_error_gold_spans * 100.0 / total_error_cnt
    print("Total: #Errors: {0}, #Overlap Errors: {1}, #Ratio of Overlap Errors: {2:.4f}"
          .format(total_error_cnt, num_overlap_error_gold_spans, ratio))


def analyse_error_overlaps(dir_path, data, dump_errors=False):
    tp, fp, fn = calc_micro_f1(data)

    print("False +ve Overlap Error Stats:")
    print_overlap_error_stats(fp, data)
    print()
    print("False -ve Overlap Error Stats:")
    print_overlap_error_stats(fn, data)

    os.makedirs(dir_path, exist_ok=True)
    tags = set(tp.keys()).union(fp.keys()).union(fn.keys())
    for tag in tags:
        outfile = os.path.join(dir_path, "{0}_auto.tsv".format(tag)) if dump_errors else None
        analyse_overlap_errors_for_tag(data, tag, verbose=True, outfile=outfile)


def analyse_oov_errors(train_data, test_data):
    tp, fp, fn = calc_micro_f1(test_data)
    train_term_vocab = set([tok[0] for sent in train_data for tok in sent])

    total_unigram_error_cnt = 0
    total_error_cnt = 0
    total_oov_unigram_error_cnt = 0
    error_dict = fn

    for tag in error_dict:
        unigram_error_cnt = 0
        oov_unigram_error_cnt = 0
        oov_error_terms = set()
        in_vocab_error_terms = set()
        for span in error_dict[tag]:
            if len(span[2]) == 1:
                unigram_error_cnt += 1
                if span[2][0] not in train_term_vocab:
                    oov_unigram_error_cnt += 1
            for term in span[2]:
                if term not in train_term_vocab:
                    oov_error_terms.add(term)
                else:
                    in_vocab_error_terms.add(term)

        total_oov_unigram_error_cnt += oov_unigram_error_cnt
        total_unigram_error_cnt += unigram_error_cnt
        total_error_cnt += len(error_dict[tag])

        print("{0}: #oov_unigram_error_terms: {1} | #unigram_error_terms: {2} | #total_errors: {3}"
              .format(tag, oov_unigram_error_cnt, unigram_error_cnt, len(error_dict[tag])))
        print("oov_error_terms({0}): {1}".format(len(oov_error_terms), oov_error_terms))
        print("in_vocab_error_terms({0}): {1}".format(len(in_vocab_error_terms), in_vocab_error_terms))
        print()

    print("TOTAL: #oov_unigram_error_terms: {0} | #unigram_error_terms: {1} | #total_errors: {2}"
          .format(total_oov_unigram_error_cnt, total_unigram_error_cnt, total_error_cnt))

    unigram_error_ratio = total_unigram_error_cnt * 100.0 / total_error_cnt
    oov_unigram_error_ratio = total_oov_unigram_error_cnt * 100.0 / total_error_cnt
    print("unigram_error_ratio: {0:.4f} | oov_unigram_error_ratio: {1:.4f}"
          .format(unigram_error_ratio, oov_unigram_error_ratio))


def get_boundary_error_ratio(data):
    gold_spans = [get_spans([[tok[0], tok[1]] for tok in sent], index) for index, sent in enumerate(data)]
    pred_spans = [get_spans([[tok[0], tok[2]] for tok in sent], index) for index, sent in enumerate(data)]
    boundary_error_cnt = 0
    exact_match_tag_error_cnt = 0
    total_error_cnt = 0
    for i in range(len(data)):
        for g_span in gold_spans[i]:
            has_error = False
            has_boundary_error = False
            for p_span in pred_spans[i]:
                if g_span == p_span:
                    has_error = False
                    break
                has_error = True
                g_start, g_end = g_span[3]
                g_tag = g_span[0]
                p_start, p_end = p_span[3]
                p_tag = p_span[0]
                if g_start == p_start and g_end == p_end:
                    exact_match_tag_error_cnt += 1
                    has_boundary_error = False
                    break
                if (g_start <= p_start <= g_end or g_start <= p_end <= g_end) and g_tag == p_tag:
                    has_boundary_error = True
            if has_error:
                total_error_cnt += 1
            if has_boundary_error:
                boundary_error_cnt += 1
    print("total: {0} | boundary: {1} | exact-match-tag-error: {2}".format(total_error_cnt, boundary_error_cnt,
                                                                           exact_match_tag_error_cnt))
    print("Boundary errors: {0:.2f}".format(boundary_error_cnt * 100.0 / (total_error_cnt + 1e-7)))
    print("Exact match - tag errors: {0:.2f}".format(exact_match_tag_error_cnt * 100.0 / (total_error_cnt + 1e-7)))


def convert_to_span_based(data):
    new_data = []
    for sent in data:
        new_sent = []
        for tup in sent:
            token, gold, pred = tup
            if gold != "O":
                gold = "{0}-ENTITY".format(gold[0])
            new_sent.append((token, gold, pred))
        new_data.append(new_sent)
    return new_data


def main(args):
    root_path = os.path.join("..", "out", args.dataset, args.model, "predictions")
    train_path = os.path.join(root_path, "train.tsv")
    dev_path = os.path.join(root_path, "dev.tsv")  # "dev"/"dev1"/"dev2" based on the mapping scheme defined in main.py
    test_path = os.path.join(root_path, "test.tsv")
    infer_path = os.path.join(root_path, "infer.tsv")

    data = dict()
    data["train"] = parse_file(train_path)
    data["dev"] = parse_file(dev_path)
    data["test"] = parse_file(test_path)
    data["infer"] = parse_file(infer_path)

    if args.span_based:
        data["train"] = convert_to_span_based(data["train"])
        data["dev"] = convert_to_span_based(data["dev"])
        data["test"] = convert_to_span_based(data["test"])

    get_boundary_error_ratio(data[args.file])
    if args.only_f1:
        calc_micro_f1(data[args.file])
    else:
        analyse_errors(data[args.file])
        analyse_error_overlaps(os.path.join(root_path, "analysis"), data[args.file], dump_errors=True)
        analyse_oov_errors(data["train"], data[args.file])


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Predictions Analyzer")
    ap.add_argument("--dataset", type=str, default="bio")
    ap.add_argument("--model", type=str, default="ner-biobert-qa4")
    ap.add_argument("--file", type=str, default="test", help="which file to evaluate (train|dev|test|infer)")
    ap.add_argument("--only_f1", dest="only_f1", action="store_true", help="set this flag to only report micro-f1")
    ap.add_argument("--span_based", dest="span_based", action="store_true", help="set this flag if using span detector")
    ap = ap.parse_args()
    main(ap)
