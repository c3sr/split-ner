from collections import defaultdict

import numpy as np


class Evaluator:

    def __init__(self, gold, predicted, tags, mask=None, ignore_tags=None, none_tag="O", pad_tag="<PAD>"):

        self.gold = gold
        self.predicted = predicted
        self.mask = mask if isinstance(mask, np.ndarray) else np.ones(self.gold.shape)
        self.tags = tags
        self.none_tag = none_tag
        self.pad_tag = pad_tag
        self.ignore_tags = ignore_tags if ignore_tags else [self.none_tag, self.pad_tag]

        self.entity_metric, self.entity_mapping = self.calc_entity_metrics()

        overall_tag_to_name_mapping = {k: v for k, v in enumerate(tags)}
        self.overall_token_metric, self.overall_token_mapping = self.calc_token_metrics(overall_tag_to_name_mapping)

        significant_tag_to_name_mapping = {k: v for k, v in enumerate(tags) if v not in self.ignore_tags}
        self.significant_token_metric, self.significant_token_mapping = self.calc_token_metrics(
            significant_tag_to_name_mapping)

    def calc_entity_metrics(self):
        bi_tag_mapping = self.generate_bi_tag_mapping()

        # entity spans are tagged with B-tag's ID
        gold_spans = self.get_spans(self.gold, bi_tag_mapping)
        predicted_spans = self.get_spans(self.predicted, bi_tag_mapping)
        entity_span_tag_to_name_mapping = {k: v[2:] for k, v in enumerate(self.tags) if v.startswith("B-")}

        entity_metric = Metric(entity_span_tag_to_name_mapping)
        for gold_sent_spans, predicted_sent_spans in zip(gold_spans, predicted_spans):
            for span in predicted_sent_spans:
                if span in gold_sent_spans:
                    entity_metric.add_tp(span)
                else:
                    entity_metric.add_fp(span)
            for span in gold_sent_spans:
                if span not in predicted_sent_spans:
                    entity_metric.add_fn(span)
        return entity_metric, entity_span_tag_to_name_mapping

    def calc_token_metrics(self, tag_to_name_mapping):
        batch_size, seq_len = self.gold.shape
        metric = Metric(tag_to_name_mapping)
        for b in range(batch_size):
            for s in range(seq_len):
                if self.mask[b, s] == 0:
                    continue
                if self.gold[b, s] == self.predicted[b, s]:
                    if self.gold[b, s] in tag_to_name_mapping:
                        metric.add_tp(Span(s, s, self.gold[b, s]))
                else:
                    if self.gold[b, s] in tag_to_name_mapping:
                        metric.add_fn(Span(s, s, self.gold[b, s]))
                    if self.predicted[b, s] in tag_to_name_mapping:
                        metric.add_fp(Span(s, s, self.predicted[b, s]))
        return metric, tag_to_name_mapping

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

    def get_spans(self, batch, bi_tag_mapping):
        batch_size, seq_len = batch.shape
        batch_spans = []
        for b in range(batch_size):
            sent_spans = []
            prev_span = None
            for s in range(seq_len):
                if self.gold[b, s] == self.pad_tag or self.mask[b, s] == 0:
                    prev_span = None
                    continue
                if batch[b, s] in bi_tag_mapping:
                    curr_span = Span(s, s, batch[b, s])
                    sent_spans.append(curr_span)
                    prev_span = curr_span
                elif prev_span:
                    if batch[b, s] == bi_tag_mapping[prev_span.tag]:
                        prev_span.end = s
                    else:
                        prev_span = None
            batch_spans.append(sent_spans)
        return batch_spans


class SpanEvaluator:
    def __init__(self, gold, predicted, tags):

        self.gold = gold
        self.predicted = predicted
        self.mapping = {tag: tag for tag in tags}
        self.entity_metric, self.entity_mapping = self.calc_entity_metrics()

    def calc_entity_metrics(self):
        entity_metric = Metric(self.mapping)
        for gold_sent_spans, predicted_sent_spans in zip(self.gold, self.predicted):
            for span in predicted_sent_spans:
                if span in gold_sent_spans:
                    entity_metric.add_tp(span)
                else:
                    entity_metric.add_fp(span)
            for span in gold_sent_spans:
                if span not in predicted_sent_spans:
                    entity_metric.add_fn(span)
        return entity_metric, self.mapping

    # TODO: Implement token-level metrics


class Span:

    def __init__(self, start, end, tag):
        self.start = start
        self.end = end
        self.tag = tag

    def __eq__(self, other):
        if not isinstance(other, Span):
            return NotImplemented
        return self.start == other.start and self.end == other.end and self.tag == other.tag

    def __repr__(self):
        return "{0} ({1}, {2})".format(self.tag, self.start, self.end)


class Metric:

    def __init__(self, mapping):
        self.mapping = mapping
        self.tp = defaultdict(list)
        self.fp = defaultdict(list)
        self.fn = defaultdict(list)

    def add_tp(self, span):
        self.tp[self.mapping[span.tag]].append(span)

    def get_tp(self, tag_name=None):
        if tag_name:
            return len(self.tp[tag_name])
        return sum(len(self.tp[self.mapping[x]]) for x in self.mapping)

    def add_fp(self, span):
        self.fp[self.mapping[span.tag]].append(span)

    def get_fp(self, tag_name=None):
        if tag_name:
            return len(self.fp[tag_name])
        return sum(len(self.fp[self.mapping[x]]) for x in self.mapping)

    def add_fn(self, span):
        self.fn[self.mapping[span.tag]].append(span)

    def get_fn(self, tag_name=None):
        if tag_name:
            return len(self.fn[tag_name])
        return sum(len(self.fn[self.mapping[x]]) for x in self.mapping)

    def accuracy(self, tag_name=None):
        tp_cnt = self.get_tp(tag_name)
        tn_cnt = tp_cnt  # assuming true-positive count to be same as true negative count
        fp_cnt = self.get_fp(tag_name)
        fn_cnt = self.get_fn(tag_name)
        tot_cnt = tp_cnt + tn_cnt + fp_cnt + fn_cnt
        if tot_cnt > 0:
            return (tp_cnt + tn_cnt) / (tot_cnt + 1e-9)
        return 0.0

    def precision(self, tag_name=None):
        tp_cnt = self.get_tp(tag_name)
        fp_cnt = self.get_fp(tag_name)
        if tp_cnt + fp_cnt > 0:
            return tp_cnt / (tp_cnt + fp_cnt + 1e-9)
        return 0.0

    def recall(self, tag_name=None):
        tp_cnt = self.get_tp(tag_name)
        fn_cnt = self.get_fn(tag_name)
        if tp_cnt + fn_cnt > 0:
            return tp_cnt / (tp_cnt + fn_cnt + 1e-9)
        return 0.0

    def f1(self, tag_name=None):
        p = self.precision(tag_name)
        r = self.recall(tag_name)
        if p + r > 0:
            return 2.0 * p * r / (p + r + 1e-9)
        return 0.0

    def macro_avg_precision(self):
        return sum([self.precision(self.mapping[x]) for x in self.mapping]) / (len(self.mapping) + 1e-9)

    def macro_avg_recall(self):
        return sum([self.recall(self.mapping[x]) for x in self.mapping]) / (len(self.mapping) + 1e-9)

    def macro_avg_f1(self):
        return sum([self.f1(self.mapping[x]) for x in self.mapping]) / (len(self.mapping) + 1e-9)

    def macro_avg_accuracy(self):
        return sum([self.accuracy(self.mapping[x]) for x in self.mapping]) / (len(self.mapping) + 1e-9)

    def micro_avg_precision(self):
        return self.precision(None)

    def micro_avg_recall(self):
        return self.recall(None)

    def micro_avg_f1(self):
        return self.f1(None)

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def report(self):
        s = "Tag\tTP\tFP\tFN\tAccuracy\tPrecision\tRecall\tF1\n"
        s += "{0}\t{1}\t{2}\t{3}\t{4:.3f}\t{5:.3f}\t{6:.3f}\t{7:.3f}\n".format(
            "OVERALL (Macro)", self.get_tp(None), self.get_fp(None), self.get_fn(None), self.macro_avg_accuracy(),
            self.macro_avg_precision(), self.macro_avg_recall(), self.macro_avg_f1())
        s += "{0}\t{1}\t{2}\t{3}\t{4:.3f}\t{5:.3f}\t{6:.3f}\t{7:.3f}\n".format(
            "OVERALL (Micro)", self.get_tp(None), self.get_fp(None), self.get_fn(None), self.micro_avg_accuracy(),
            self.micro_avg_precision(), self.micro_avg_recall(), self.micro_avg_f1())
        for k in self.mapping:
            tag = self.mapping[k]
            s += "{0}\t{1}\t{2}\t{3}\t{4:.3f}\t{5:.3f}\t{6:.3f}\t{7:.3f}\n".format(
                tag, self.get_tp(tag), self.get_fp(tag), self.get_fn(tag), self.accuracy(tag), self.precision(tag),
                self.recall(tag), self.f1(tag))

        return s
