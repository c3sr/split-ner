import argparse
import re

import spacy
from spacy.tokens.doc import Doc
from torch.utils.data import Dataset
from transformers import HfArgumentParser, AutoTokenizer

from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataset
from secner.utils.general import Token, set_all_seeds, BertToken, parse_config, setup_logging, Context


class NerQADataset(Dataset):

    def __init__(self, args: AdditionalArguments, corpus_type):
        super(NerQADataset, self).__init__()
        self.args = args
        self.corpus_type = corpus_type
        self.corpus_path = self.set_corpus_path()
        self.pos_tag_vocab = NerDataset.parse_aux_tag_vocab(self.args.pos_tag_vocab_path, self.args.none_tag,
                                                            self.args.use_pos_tag)
        self.dep_tag_vocab = NerDataset.parse_aux_tag_vocab(self.args.dep_tag_vocab_path, self.args.none_tag,
                                                            self.args.use_dep_tag)

        self.tag_to_text_mapping = self.parse_tag_names()

        if self.args.use_pos_tag or self.args.use_dep_tag:
            self.nlp = spacy.load("en_core_web_sm")
            self.tokenizer_map = dict()
            self.nlp.tokenizer = lambda x: Doc(self.nlp.vocab, self.tokenizer_map[x])

        self.contexts = []
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        self.bert_start_token, self.bert_first_sep_token, self.bert_second_sep_token = \
            NerDataset.get_bert_special_tokens(self.tokenizer, self.args.none_tag)
        self.tokenizer_cache = dict()
        self.sentences = NerDataset.read_dataset(self.corpus_path, self.args)
        self.filter_tags()
        self.split_tags()
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

    def filter_tags(self):
        if self.args.filter_tags is None:
            return
        permissible_tags = [self.args.none_tag]
        permissible_tags.extend(self.args.filter_tags)
        permissible_tags = set(permissible_tags)
        remove_tags = set(self.tag_to_text_mapping.keys()) - permissible_tags
        for tag in remove_tags:
            del self.tag_to_text_mapping[tag]

        for sent in self.sentences:
            for tok in sent.tokens:
                new_tags = tok.tags
                for tag in tok.tags:
                    if tag[2:] not in permissible_tags:
                        new_tags.remove(tag)
                if len(new_tags) == 0:
                    new_tags.append(self.args.none_tag)
                tok.tags = new_tags

    def split_tags(self):
        if not self.args.split_tags:
            return
        if self.args.dataset_dir == "bio":
            self.tag_to_text_mapping["Symbolic_simple_chemical"] = "symbolic simple chemical"
            for sent in self.sentences:
                spans = NerDataset.get_spans(sent)
                for sp in spans["Simple_chemical"]:
                    mention = " ".join([sent.tokens[i].text for i in range(sp.start, sp.end + 1)])
                    if re.search(r"[^A-Za-z0-9]|\d", mention):
                        k = sent.tokens[sp.start].tags.index("B-Simple_chemical")
                        sent.tokens[sp.start].tags[k] = "B-Symbolic_simple_chemical"
                        for index in range(sp.start + 1, sp.end + 1):
                            k = sent.tokens[index].tags.index("I-Simple_chemical")
                            sent.tokens[index].tags[k] = "I-Symbolic_simple_chemical"

    def parse_dataset(self):
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
        bert_head_mask = [tok.is_head for tok in context.bert_tokens]
        bert_token_text = [tok.token.text for tok in context.bert_tokens]
        bert_sub_token_text = [tok.sub_text for tok in context.bert_tokens]
        bert_token_pos = [self.pos_tag_vocab.index(tok.token.pos_tag) for tok in
                          context.bert_tokens] if self.args.use_pos_tag else []
        bert_token_dep = [self.dep_tag_vocab.index(tok.token.dep_tag) for tok in
                          context.bert_tokens] if self.args.use_dep_tag else []
        bert_tag_ids = [NerQADataset.get_tag_index(tok.token.tags[0], self.args.none_tag) for tok in
                        context.bert_tokens]

        print("labels="+str([tok.token.tags[0] for tok in context.bert_tokens]))
        print("label_ids="+str(bert_tag_ids))

        return {"input_ids": bert_token_ids,
                "token_type_ids": bert_token_type_ids,
                "head_mask": bert_head_mask,
                "text": bert_token_text,
                "sub_text": bert_sub_token_text,
                "pos_tag": bert_token_pos,
                "dep_tag": bert_token_dep,
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
        if text_tag == "S":
            return 4
        # should never occur
        return -100

    def tokenize_with_cache(self, text):
        if text not in self.tokenizer_cache:
            self.tokenizer_cache[text] = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return self.tokenizer_cache[text]

    def get_tag_query_text(self, tag):
        if self.args.detect_spans:
            if self.args.query_type == "question3":
                tag_text = "named entities"
            elif self.args.query_type == "question4":
                tag_text = "important entity spans"
            else:
                tag_text = "entity"
        else:
            tag_text = self.tag_to_text_mapping[tag]
        if self.args.query_type == "question":
            return "What is the {0} mentioned in the text ?".format(tag_text)
        if self.args.query_type == "question2":
            return "Where is the {0} mentioned in the text ?".format(tag_text)
        if self.args.query_type == "question3":
            return "Find {0} in the following text .".format(tag_text)
        if self.args.query_type == "question4":
            return "Extract {0} from the following text .".format(tag_text)
        return tag_text

    def prep_context(self, sentence, tag):
        tag_text = self.get_tag_query_text(tag)
        # query
        bert_query_tokens = []
        query_tokens = tag_text.split()
        doc = None
        if self.args.use_pos_tag or self.args.use_dep_tag:
            self.tokenizer_map[tag_text] = query_tokens
            doc = self.nlp(tag_text)
        for index, word in enumerate(query_tokens):
            out = self.tokenize_with_cache(word)
            pos_tag = doc[index].tag_ if self.args.use_pos_tag else None
            dep_tag = doc[index].dep_ if self.args.use_dep_tag else None
            for i in range(len(out["input_ids"])):
                bert_token = Token(word, [self.args.none_tag], offset=index, pos_tag=pos_tag, dep_tag=dep_tag)
                tup = out["offset_mapping"][i]
                sub_text = word[tup[0]:tup[1]]
                bert_query_tokens.append(BertToken(bert_id=out["input_ids"][i], sub_text=sub_text, token_type=0,
                                                   token=bert_token, is_head=(i == 0)))

        # helper sentence
        bert_helper_sent_tokens = []
        if self.args.add_qa_helper_sentence:
            q = len(query_tokens)
            for tok in sentence.tokens:
                token_tags = [t[2:] for t in tok.tags]
                if self.corpus_type != "train" or tag in token_tags or self.args.none_tag in token_tags:
                    helper_text = tok.text
                elif "B-{0}".format(tag) in tok.tags:
                    helper_text = self.tag_to_text_mapping[tag]
                else:
                    continue
                out = self.tokenize_with_cache(helper_text)
                for i in range(len(out["input_ids"])):
                    bert_token = Token(helper_text, [self.args.none_tag], q + tok.offset, tok.pos_tag, tok.dep_tag,
                                       tok.guidance_tag)
                    tup = out["offset_mapping"][i]
                    sub_text = helper_text[tup[0]:tup[1]]
                    bert_helper_sent_tokens.append(BertToken(bert_id=out["input_ids"][i], sub_text=sub_text,
                                                             token_type=0, token=bert_token, is_head=(i == 0)))

        # sentence
        bert_sent_tokens = []
        for tok in sentence.tokens:
            token_tags = [t[2:] for t in tok.tags]
            if tag in token_tags:
                new_tag = tok.tags[token_tags.index(tag)][0]
            else:
                new_tag = self.args.none_tag
            out = self.tokenize_with_cache(tok.text)
            for i in range(len(out["input_ids"])):
                if self.args.use_head_mask:
                    bert_tag = new_tag
                else:
                    bert_tag = new_tag if i == 0 or new_tag != "B" else "I"
                bert_token = Token(tok.text, [bert_tag], tok.offset, tok.pos_tag, tok.dep_tag, tok.guidance_tag)
                tup = out["offset_mapping"][i]
                sub_text = tok.text[tup[0]:tup[1]]
                bert_sent_tokens.append(BertToken(bert_id=out["input_ids"][i], sub_text=sub_text, token_type=1,
                                                  token=bert_token, is_head=(i == 0)))

        if self.args.num_labels == 2:
            # BO tagging scheme
            for i in range(len(bert_sent_tokens)):
                if bert_sent_tokens[i].token.tags[0] == "I":
                    if not self.args.use_head_mask or bert_sent_tokens[i].is_head:
                        bert_sent_tokens[i].token.tags[0] = "B"

        elif self.args.num_labels > 3:
            # BIOE tagging scheme
            is_end_token = False
            for i in range(len(bert_sent_tokens) - 1, 0, -1):
                if bert_sent_tokens[i].token.tags[0] == "I":
                    if is_end_token:
                        if not self.args.use_head_mask or bert_sent_tokens[i].is_head:
                            bert_sent_tokens[i].token.tags[0] = "E"
                            is_end_token = False
                else:
                    is_end_token = True

            if self.args.num_labels == 5:
                # BIOES tagging scheme
                if self.args.use_head_mask:
                    mention_length = 0
                    mention_index = -1
                    for i in range(len(bert_sent_tokens)):
                        if not bert_sent_tokens[i].is_head:
                            continue
                        if bert_sent_tokens[i].token.tags[0] == "B":
                            if mention_length == 1:
                                bert_sent_tokens[mention_index].token.tags[0] = "S"
                            mention_length = 1
                            mention_index = i
                        elif bert_sent_tokens[i].token.tags[0] in ["I", "E"]:
                            mention_length += 1
                        elif mention_length == 1:
                            bert_sent_tokens[mention_index].token.tags[0] = "S"
                            mention_length = 0
                            mention_index = -1
                    if mention_length == 1:
                        bert_sent_tokens[mention_index].token.tags[0] = "S"
                else:
                    for i in range(len(bert_sent_tokens)):
                        if bert_sent_tokens[i].token.tags[0] == "B" and i + 1 < len(bert_sent_tokens) and \
                                bert_sent_tokens[i + 1].token.tags[0] not in ["I", "E"]:
                            bert_sent_tokens[i].token.tags[0] = "S"

        bert_tokens = [self.bert_start_token]
        bert_tokens.extend(bert_query_tokens)
        bert_tokens.extend(bert_helper_sent_tokens)
        bert_tokens.append(self.bert_first_sep_token)
        if self.args.model_mode == "roberta_std":
            bert_tokens.append(self.bert_first_sep_token)  # check
        bert_tokens.extend(bert_sent_tokens)
        bert_tokens = bert_tokens[:self.args.max_seq_len - 1]
        bert_tokens.append(self.bert_second_sep_token)
        return Context(sentence, tag, tag_text, bert_tokens)

    def prep_context_span(self, sentence):
        tag_text = self.get_tag_query_text(None)
        # query
        bert_query_tokens = []
        query_tokens = tag_text.split()
        doc = None
        if self.args.use_pos_tag or self.args.use_dep_tag:
            self.tokenizer_map[tag_text] = query_tokens
            doc = self.nlp(tag_text)
        for index, word in enumerate(query_tokens):
            out = self.tokenize_with_cache(word)
            pos_tag = doc[index].tag_ if self.args.use_pos_tag else None
            dep_tag = doc[index].dep_ if self.args.use_dep_tag else None
            for i in range(len(out["input_ids"])):
                bert_token = Token(word, [self.args.none_tag], offset=index, pos_tag=pos_tag, dep_tag=dep_tag)
                tup = out["offset_mapping"][i]
                sub_text = word[tup[0]:tup[1]]
                bert_query_tokens.append(BertToken(bert_id=out["input_ids"][i], sub_text=sub_text, token_type=0,
                                                   token=bert_token, is_head=(i == 0)))

        # sentence
        bert_sent_tokens = []
        for tok in sentence.tokens:
            token_tags = [t[2:] for t in tok.tags]
            if token_tags != [self.args.none_tag]:
                # TODO: This needs to be corrected for nested entity cases
                new_tag = tok.tags[0][0]
            else:
                new_tag = self.args.none_tag
            out = self.tokenize_with_cache(tok.text)
            for i in range(len(out["input_ids"])):
                bert_tag = new_tag if i == 0 or new_tag != "B" else "I"
                bert_token = Token(tok.text, [bert_tag], tok.offset, tok.pos_tag, tok.dep_tag, tok.guidance_tag)
                tup = out["offset_mapping"][i]
                sub_text = tok.text[tup[0]:tup[1]]
                bert_sent_tokens.append(BertToken(bert_id=out["input_ids"][i], sub_text=sub_text, token_type=1,
                                                  token=bert_token, is_head=(i == 0)))

        if self.args.num_labels == 2:
            # BO tagging scheme
            for i in range(len(bert_sent_tokens)):
                if bert_sent_tokens[i].token.tags[0] == "I":
                    if not self.args.use_head_mask or bert_sent_tokens[i].is_head:
                        bert_sent_tokens[i].token.tags[0] = "B"

        elif self.args.num_labels > 3:
            # BIOE tagging scheme
            is_end_token = False
            for i in range(len(bert_sent_tokens) - 1, 0, -1):
                if bert_sent_tokens[i].token.tags[0] == "I":
                    if is_end_token:
                        if not self.args.use_head_mask or bert_sent_tokens[i].is_head:
                            bert_sent_tokens[i].token.tags[0] = "E"
                            is_end_token = False
                else:
                    is_end_token = True

            if self.args.num_labels == 5:
                # BIOES tagging scheme
                if self.args.use_head_mask:
                    mention_length = 0
                    mention_index = -1
                    for i in range(len(bert_sent_tokens)):
                        if not bert_sent_tokens[i].is_head:
                            continue
                        if bert_sent_tokens[i].token.tags[0] == "B":
                            if mention_length == 1:
                                bert_sent_tokens[mention_index].token.tags[0] = "S"
                            mention_length = 1
                            mention_index = i
                        elif bert_sent_tokens[i].token.tags[0] in ["I", "E"]:
                            mention_length += 1
                        elif mention_length == 1:
                            bert_sent_tokens[mention_index].token.tags[0] = "S"
                            mention_length = 0
                            mention_index = -1
                    if mention_length == 1:
                        bert_sent_tokens[mention_index].token.tags[0] = "S"
                else:
                    for i in range(len(bert_sent_tokens)):
                        if bert_sent_tokens[i].token.tags[0] == "B" and i + 1 < len(bert_sent_tokens) and \
                                bert_sent_tokens[i + 1].token.tags[0] not in ["I", "E"]:
                            bert_sent_tokens[i].token.tags[0] = "S"

        bert_tokens = [self.bert_start_token]
        bert_tokens.extend(bert_query_tokens)
        bert_tokens.append(self.bert_first_sep_token)
        bert_tokens.extend(bert_sent_tokens)
        bert_tokens = bert_tokens[:self.args.max_seq_len - 1]
        bert_tokens.append(self.bert_second_sep_token)
        return Context(sentence, "ENTITY", tag_text, bert_tokens)

    def process_sentence(self, sentence):
        if self.args.detect_spans:
            self.contexts.append(self.prep_context_span(sentence))
        else:
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
