from collections import defaultdict

import numpy as np


class EvaluatorSpan:

    def __init__(self, gold, predicted, tags):
        self.gold = gold.tolist() if isinstance(gold, np.ndarray) else gold
        self.predicted = predicted.tolist() if isinstance(predicted, np.ndarray) else predicted
        self.tags = tags
        self.entity_metric = self.calc_entity_metrics()

    def calc_entity_metrics(self):
        metric = SpanClassificationMetric(self.tags)
        for g, p in zip(self.gold, self.predicted):
            if g == -100:
                # TODO: Currently ignoring any explicit "false positive span inputs" from model metric calculations
                continue
            if g == p:
                metric.add_tp(self.tags[g])
            else:
                metric.add_fp(self.tags[p])
                metric.add_fn(self.tags[g])
        return metric


class SpanClassificationMetric:

    def __init__(self, tags):
        self.tags = tags
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

    def add_tp(self, tag):
        self.tp[tag] += 1

    def get_tp(self, tag_name=None):
        if tag_name:
            return self.tp[tag_name]
        return sum(self.tp[tag] for tag in self.tags)

    def add_fp(self, tag):
        self.fp[tag] += 1

    def get_fp(self, tag_name=None):
        if tag_name:
            return self.fp[tag_name]
        return sum(self.fp[tag] for tag in self.tags)

    def add_fn(self, tag):
        self.fn[tag] += 1

    def get_fn(self, tag_name=None):
        if tag_name:
            return self.fn[tag_name]
        return sum(self.fn[tag] for tag in self.tags)

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
