import argparse

from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataset
from secner.utils.general import Token, set_all_seeds, BertToken, parse_config, setup_logging, Context
from torch.utils.data import Dataset
from transformers import HfArgumentParser, AutoTokenizer


class NerQADataset(Dataset):

    def __init__(self, args: AdditionalArguments, corpus_type):
        super(NerQADataset, self).__init__()
        self.args = args
        self.corpus_type = corpus_type
        self.corpus_path = self.set_corpus_path()

        self.tag_to_text_mapping = self.parse_tag_names()

        self.contexts = []
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        self.bert_start_token, self.bert_mid_sep_token, self.bert_end_token = NerDataset.get_bert_special_tokens(
            self.tokenizer, self.args.none_tag)
        self.tokenizer_cache = dict()
        self.parse_dataset()

    def set_corpus_path(self):
        if self.corpus_type == "train":
            return self.args.train_path
        if self.corpus_type == "dev":
            return self.args.dev_path
        if self.corpus_type == "test":
            return self.args.test_path
        return None

    def parse_tag_names(self):
        tag_to_text_mapping = dict()
        with open(self.args.tag_names_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = line.split("\t")
                    tag_to_text_mapping[s[0]] = s[1]
        return tag_to_text_mapping

    def parse_dataset(self):
        sentences = NerDataset.read_dataset(self.corpus_path, self.args)
        for sentence in sentences:
            self.process_sentence(sentence)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        context = self.contexts[index]
        bert_token_ids = [tok.bert_id for tok in context.bert_tokens]
        bert_token_type_ids = [tok.token_type for tok in context.bert_tokens]
        bert_token_text = [tok.token.text for tok in context.bert_tokens]
        bert_tag_ids = [NerQADataset.get_tag_index(tok.token.tag, self.args.none_tag) for tok in context.bert_tokens]

        return {"input_ids": bert_token_ids,
                "token_type_ids": bert_token_type_ids,
                "text": bert_token_text,
                "labels": bert_tag_ids}

    @staticmethod
    def get_tag_index(text_tag, none_tag):
        if text_tag == none_tag:
            return 0
        if text_tag == "B":
            return 1
        if text_tag == "I":
            return 2
        if text_tag == "E":
            return 3
        # should never occur
        return -100

    def tokenize_with_cache(self, text):
        if text not in self.tokenizer_cache:
            self.tokenizer_cache[text] = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer_cache[text]

    def get_tag_query_text(self, tag):
        tag_text = self.tag_to_text_mapping[tag]
        if self.args.query_type == "question":
            return "What is the {0} mentioned in the text ?".format(tag_text)
        return tag_text

    def prep_context(self, sentence, tag):
        tag_text = self.get_tag_query_text(tag)
        # query
        bert_query_tokens = []
        query_tokens = tag_text.split()
        for index, word in enumerate(query_tokens):
            bert_ids = self.tokenize_with_cache(word)
            for i in range(len(bert_ids)):
                bert_token = Token(word, self.args.none_tag, index)
                bert_query_tokens.append(BertToken(bert_id=bert_ids[i], token_type=0, token=bert_token))

        # helper sentence
        bert_helper_sent_tokens = []
        if self.args.add_qa_helper_sentence:
            q = len(query_tokens)
            for tok in sentence.tokens:
                if tok.tag[2:] == tag or tok.tag == self.args.none_tag:
                    helper_text = tok.text
                elif tok.tag.startswith("B-"):
                    helper_text = self.tag_to_text_mapping[tok.tag[2:]]
                else:
                    continue
                bert_ids = self.tokenize_with_cache(helper_text)
                for i in range(len(bert_ids)):
                    bert_token = Token(helper_text, self.args.none_tag, q + tok.offset, tok.pos_tag, tok.dep_tag,
                                       tok.guidance_tag)
                    bert_helper_sent_tokens.append(BertToken(bert_id=bert_ids[i], token_type=0, token=bert_token))

        # sentence
        bert_sent_tokens = []
        for tok in sentence.tokens:
            new_tag = tok.tag[0] if tok.tag[2:] == tag else self.args.none_tag
            bert_ids = self.tokenize_with_cache(tok.text)
            for i in range(len(bert_ids)):
                bert_tag = new_tag if i == 0 or new_tag != "B" else "I"
                bert_token = Token(tok.text, bert_tag, tok.offset, tok.pos_tag, tok.dep_tag, tok.guidance_tag)
                bert_sent_tokens.append(BertToken(bert_id=bert_ids[i], token_type=1, token=bert_token))

        if self.args.num_labels == 2:
            # BO tagging scheme
            for i in range(len(bert_sent_tokens)):
                if bert_sent_tokens[i].token.tag == "I":
                    bert_sent_tokens[i].token.tag = "B"

        elif self.args.num_labels == 4:
            # BIOE tagging scheme
            is_end_token = False
            for i in range(len(bert_sent_tokens) - 1, 0, -1):
                if bert_sent_tokens[i].token.tag == "I":
                    if is_end_token:
                        bert_sent_tokens[i].token.tag = "E"
                        is_end_token = False
                else:
                    is_end_token = True

        bert_tokens = [self.bert_start_token]
        bert_tokens.extend(bert_query_tokens)
        bert_tokens.extend(bert_helper_sent_tokens)
        bert_tokens.append(self.bert_mid_sep_token)
        bert_tokens.extend(bert_sent_tokens)
        bert_tokens.append(self.bert_end_token)
        return Context(sentence, tag, tag_text, bert_tokens)

    def process_sentence(self, sentence):
        for tag in self.tag_to_text_mapping.keys():
            self.contexts.append(self.prep_context(sentence, tag))


def main(args):
    setup_logging()
    parser = HfArgumentParser([AdditionalArguments])
    additional_args = parse_config(parser, args.config)[0]
    set_all_seeds(42)
    dataset = NerQADataset(additional_args, corpus_type="test")
    print(len(dataset))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QA Dataset Runner")
    ap.add_argument("--config", default="config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
