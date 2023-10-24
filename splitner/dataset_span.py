import argparse
import logging

import torch
from attr import dataclass
from torch.utils.data import Dataset
from transformers import HfArgumentParser, AutoTokenizer

from splitner.additional_args import AdditionalArguments
from splitner.dataset import NerDataset
from splitner.utils.general import Token, set_all_seeds, BertToken, parse_config, setup_logging, Context, Sentence, \
    PairSpan

logger = logging.getLogger(__name__)


class NerSpanDataset(Dataset):

    def __init__(self, args: AdditionalArguments, corpus_type):
        super(NerSpanDataset, self).__init__()
        self.args = args
        self.corpus_type = corpus_type  # is also used as  final predictions filename
        self.corpus_path = self.set_corpus_path()
        self.tag_vocab = NerSpanDataset.parse_tag_vocab(self.args.tag_vocab_path)

        self.contexts = []
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        self.bert_start_token, self.bert_first_sep_token, self.bert_second_sep_token = \
            NerDataset.get_bert_special_tokens(self.tokenizer, self.args.none_tag)
        self.tokenizer_cache = dict()
        self.sentences = []
        self.parse_dataset()

    @staticmethod
    def parse_tag_vocab(file_path):
        tag_vocab = []
        # only TAG from B-TAG, I-TAG is included, NONE and PAD are also ignored
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and (line.startswith("B-") or line.startswith("I-")) and line[2:] not in tag_vocab:
                    tag_vocab.append(line[2:])
        return tag_vocab

    def set_corpus_path(self):
        if self.corpus_type == "train":
            return self.args.train_path
        if self.corpus_type == "dev":
            return self.args.dev_path
        if self.corpus_type == "test":
            return self.args.test_path
        return None

    def parse_dataset(self):
        self.sentences = NerDataset.read_dataset(self.corpus_path, self.args)
        for sentence in self.sentences:
            self.process_sentence(sentence)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        context = self.contexts[index]
        bert_token_ids = [tok.bert_id for tok in context.bert_tokens]
        if self.args.model_mode == "roberta_std":
            bert_token_type_ids = [self.tokenizer.pad_token_type_id for _ in context.bert_tokens]
        else:
            bert_token_type_ids = [tok.token_type for tok in context.bert_tokens]
        # TODO: Needs to be handled if working with nested entities
        tag = context.sentence.tokens[context.mention_span.start].tags[0][2:]
        bert_tag_id = self.tag_vocab.index(tag) if tag in self.tag_vocab else -100

        return {"input_ids": bert_token_ids,
                "token_type_ids": bert_token_type_ids,
                "labels": bert_tag_id}

    def tokenize_with_cache(self, text):
        if text not in self.tokenizer_cache:
            self.tokenizer_cache[text] = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return self.tokenizer_cache[text]

    def get_mention_query_text(self, mention):
        if self.args.query_type == "question":
            return "What is {0} ?".format(mention)
        if self.args.query_type == "question2":
            return "Classify {0} .".format(mention)
        raise NotImplementedError

    def prep_context(self, sentence: Sentence, mention_span: PairSpan):
        # sentence
        bert_sent_tokens = []
        for index, tok in enumerate(sentence.tokens):
            out = self.tokenize_with_cache(tok.text)
            for i in range(len(out["input_ids"])):
                if index < mention_span.start or index > mention_span.end:
                    bert_tag = self.args.none_tag
                elif tok.tags[0].startswith("B-") and i == 0:
                    # considering first tag as the main working tag for the token
                    # TODO: needs to be handled if working with nested entities
                    bert_tag = "B-{0}".format(tok.tags[0][2:])
                else:
                    bert_tag = "I-{0}".format(tok.tags[0][2:])
                bert_token = Token(tok.text, [bert_tag], tok.offset, tok.pos_tag, tok.dep_tag, tok.guidance_tag)
                tup = out["offset_mapping"][i]
                sub_text = tok.text[tup[0]:tup[1]]
                bert_sent_tokens.append(BertToken(bert_id=out["input_ids"][i], sub_text=sub_text, token_type=0,
                                                  token=bert_token, is_head=(i == 0)))

        # query
        bert_query_tokens = []
        mention = " ".join([sentence.tokens[i].text for i in range(mention_span.start, mention_span.end + 1)])
        query_tokens = self.get_mention_query_text(mention).split()
        for index, word in enumerate(query_tokens):
            out = self.tokenize_with_cache(word)
            for i in range(len(out["input_ids"])):
                bert_token = Token(word, [self.args.none_tag], offset=index)
                tup = out["offset_mapping"][i]
                sub_text = word[tup[0]:tup[1]]
                bert_query_tokens.append(BertToken(bert_id=out["input_ids"][i], sub_text=sub_text, token_type=1,
                                                   token=bert_token, is_head=(i == 0)))

        bert_tokens = [self.bert_start_token]
        bert_tokens.extend(bert_sent_tokens)
        bert_tokens.append(self.bert_first_sep_token)
        if self.args.model_mode == "roberta_std":
            bert_tokens.append(self.bert_first_sep_token)
        bert_tokens = bert_tokens[:self.args.max_seq_len - len(bert_query_tokens) - 1]
        bert_tokens.extend(bert_query_tokens)
        bert_tokens.append(self.bert_second_sep_token)
        return Context(sentence, None, None, bert_tokens, mention_span)

    def process_sentence(self, sentence):
        spans = NerDataset.get_spans(sentence)
        for tag in spans.keys():
            for mention_span in spans[tag]:
                self.contexts.append(self.prep_context(sentence, mention_span))


class NerInferSpanDataset(NerSpanDataset):

    def __init__(self, args: AdditionalArguments):
        infer_prefix = args.infer_out_path.split(".tsv")[0]
        super(NerInferSpanDataset, self).__init__(args, corpus_type=infer_prefix)

    def parse_dataset(self):
        spans = self.parse_infer_file()

        if self.args.debug_mode:
            self.sentences = self.sentences[:10]
            spans = spans[:10]

        for i in range(len(self.sentences)):
            for sp in spans[i]:
                self.contexts.append(self.prep_context(self.sentences[i], sp))

    def parse_infer_file(self):
        sent_spans = []
        with open(self.args.infer_inp_path, "r", encoding="utf-8") as f:
            tokens = []
            spans = []
            offset = 0
            continue_span = False
            for line in f:
                line = line.strip()
                if line:
                    s = line.split("\t")
                    tokens.append(Token(text=s[0], tags=[s[1]], offset=offset))

                    if s[2].startswith("B-"):
                        spans.append(PairSpan(offset, offset))
                        continue_span = True
                    elif s[2].startswith("I-") and continue_span:
                        spans[-1].end = offset
                    else:
                        continue_span = False

                    offset += 1
                else:
                    self.sentences.append(Sentence(tokens))
                    sent_spans.append(spans)
                    tokens = []
                    spans = []
                    offset = 0
                    continue_span = False

            if len(tokens) > 0:
                self.sentences.append(Sentence(tokens))
                sent_spans.append(spans)

        return sent_spans


@dataclass
class NerSpanDataCollator:
    args: AdditionalArguments

    def __call__(self, features):
        # post-padding
        max_len = max(len(entry["input_ids"]) for entry in features)
        batch = dict()

        # input_ids
        if "input_ids" in features[0]:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["input_ids"])
                entry.append(torch.tensor(features[i]["input_ids"] + [0] * pad_len))
            batch["input_ids"] = torch.stack(entry)

        # attention_mask
        entry = []
        for i in range(len(features)):
            good_len = len(features[i]["input_ids"])
            pad_len = max_len - good_len
            entry.append(torch.tensor([1] * good_len + [0] * pad_len))
        batch["attention_mask"] = torch.stack(entry)

        # token_type_ids
        if "token_type_ids" in features[0]:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["token_type_ids"])
                entry.append(torch.tensor(features[i]["token_type_ids"] + [0] * pad_len))
            batch["token_type_ids"] = torch.stack(entry)

        # labels
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.int64)

        return batch


def main(args):
    setup_logging()
    parser = HfArgumentParser([AdditionalArguments])
    additional_args = parse_config(parser, args.config)[0]
    set_all_seeds(42)
    dataset = NerSpanDataset(additional_args, corpus_type="test")
    print(len(dataset))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Span Classification Dataset Runner")
    ap.add_argument("--config", default="config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
