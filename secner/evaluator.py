from collections import defaultdict

import numpy as np


class Evaluator:

    def __init__(self, gold, predicted, tags):
        self.gold = gold.tolist() if isinstance(gold, np.ndarray) else gold
        self.predicted = predicted.tolist() if isinstance(predicted, np.ndarray) else predicted
        self.tags = tags
        self.b_to_i = self.generate_bi_tag_mapping()
        self.i_to_b = {v: k for k, v in self.b_to_i.items()}
        self.b_to_tag = {k: v[2:] for k, v in enumerate(self.tags) if v.startswith("B-")}

        self.gold_entity_spans = self.get_spans(self.gold)
        self.predicted_entity_spans = self.get_spans(self.predicted)
        self.entity_metric = self.calc_entity_metrics()

    def calc_entity_metrics(self):
        entity_metric = Metric([tag[2:] for tag in self.tags if tag.startswith("B-")])
        for gold_sent_spans, predicted_sent_spans in zip(self.gold_entity_spans, self.predicted_entity_spans):
            for span in predicted_sent_spans:
                if span in gold_sent_spans:
                    entity_metric.add_tp(span)
                else:
                    entity_metric.add_fp(span)
            for span in gold_sent_spans:
                if span not in predicted_sent_spans:
                    entity_metric.add_fn(span)
        return entity_metric

    def generate_bi_tag_mapping(self):
        tag_to_index = dict()
        for index, tag_text in enumerate(self.tags):
            if tag_text.startswith("B-"):
                tag_to_index[tag_text[2:]] = index

        next_tag = dict()
        for index, tag_text in enumerate(self.tags):
            if tag_text.startswith("I-"):
                next_tag[tag_to_index[tag_text[2:]]] = index

        return next_tag

    def get_spans(self, batch):
        batch_spans = []
        for sent_index in range(len(batch)):
            sent_spans = []
            prev_span = None
            for tok_index in range(len(batch[sent_index])):
                if self.gold[sent_index][tok_index] == -100:
                    prev_span = None
                    continue
                if batch[sent_index][tok_index] in self.b_to_i:
                    tag = self.b_to_tag[batch[sent_index][tok_index]]
                    curr_span = Span(sent_index, tok_index, tok_index, tag)
                    sent_spans.append(curr_span)
                    prev_span = curr_span
                elif prev_span and batch[sent_index][tok_index] in self.i_to_b:
                    tag = self.b_to_tag[self.i_to_b[batch[sent_index][tok_index]]]
                    if tag == prev_span.tag:
                        prev_span.end = tok_index
                    else:
                        prev_span = None
                else:
                    prev_span = None
            batch_spans.append(sent_spans)
        return batch_spans


class Span:

    def __init__(self, sent, start, end, tag):
        self.sent = sent
        self.start = start
        self.end = end
        self.tag = tag

    def __eq__(self, other):
        if not isinstance(other, Span):
            return NotImplemented
        return self.sent == other.sent and self.start == other.start and self.end == other.end and self.tag == other.tag

    def __repr__(self):
        return "{0} (sent: {1}, start: {2}, end: {3})".format(self.tag, self.sent, self.start, self.end)


class Metric:

    def __init__(self, tags):
        self.tags = tags
        self.tp = defaultdict(list)
        self.fp = defaultdict(list)
        self.fn = defaultdict(list)

    def add_tp(self, span):
        self.tp[span.tag].append(span)

    def get_tp(self, tag_name=None):
        if tag_name:
            return len(self.tp[tag_name])
        return sum(len(self.tp[tag]) for tag in self.tags)

    def add_fp(self, span):
        self.fp[span.tag].append(span)

    def get_fp(self, tag_name=None):
        if tag_name:
            return len(self.fp[tag_name])
        return sum(len(self.fp[tag]) for tag in self.tags)

    def add_fn(self, span):
        self.fn[span.tag].append(span)

    def get_fn(self, tag_name=None):
        if tag_name:
            return len(self.fn[tag_name])
        return sum(len(self.fn[tag]) for tag in self.tags)

    def precision(self, tag_name=None):
        tp_cnt = self.get_tp(tag_name)
        fp_cnt = self.get_fp(tag_name)
        return tp_cnt / (tp_cnt + fp_cnt + 1e-9)

    def recall(self, tag_name=None):
        tp_cnt = self.get_tp(tag_name)
        fn_cnt = self.get_fn(tag_name)
        return tp_cnt / (tp_cnt + fn_cnt + 1e-9)

    def f1(self, tag_name=None):
        p = self.precision(tag_name)
        r = self.recall(tag_name)
        return 2.0 * p * r / (p + r + 1e-9)

    def macro_avg_precision(self):
        return sum([self.precision(tag) for tag in self.tags]) / (len(self.tags) + 1e-9)

    def macro_avg_recall(self):
        return sum([self.recall(tag) for tag in self.tags]) / (len(self.tags) + 1e-9)

    def macro_avg_f1(self):
        return sum([self.f1(tag) for tag in self.tags]) / (len(self.tags) + 1e-9)

    def micro_avg_precision(self):
        return self.precision(None)

    def micro_avg_recall(self):
        return self.recall(None)

    def micro_avg_f1(self):
        return self.f1(None)

    def report(self):
        s = "Tag\tTP\tFP\tFN\tPrecision(%)\tRecall(%)\tF1(%)\n"
        s += "{0}\t{1}\t{2}\t{3}\t{4:.3f}\t{5:.3f}\t{6:.3f}\n".format(
            "Macro", self.get_tp(None), self.get_fp(None), self.get_fn(None),
            100.0 * self.macro_avg_precision(), 100.0 * self.macro_avg_recall(), 100.0 * self.macro_avg_f1())
        s += "{0}\t{1}\t{2}\t{3}\t{4:.3f}\t{5:.3f}\t{6:.3f}\n".format(
            "Micro", self.get_tp(None), self.get_fp(None), self.get_fn(None),
            100.0 * self.micro_avg_precision(), 100.0 * self.micro_avg_recall(), 100.0 * self.micro_avg_f1())
        # for tag in self.tags:
        #     s += "{0}\t{1}\t{2}\t{3}\t{4:.3f}\t{5:.3f}\t{6:.3f}\n".format(
        #         tag, self.get_tp(tag), self.get_fp(tag), self.get_fn(tag),
        #         100.0 * self.precision(tag), 100.0 * self.recall(tag), 100.0 * self.f1(tag))

        return s
