import argparse
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import HfArgumentParser

from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataset
from secner.utils.general import set_all_seeds, parse_config, setup_logging


class NerCharDataset(Dataset):

    def __init__(self, args: AdditionalArguments, corpus_type):
        super(NerCharDataset, self).__init__()
        self.args = args
        self.corpus_type = corpus_type
        self.corpus_path = self.set_corpus_path()

        self.tag_vocab = []
        self.parse_tag_vocab()

        self.sentences = NerDataset.read_dataset(self.corpus_path, self.args)

    def set_corpus_path(self):
        if self.corpus_type == "train":
            return self.args.train_path
        if self.corpus_type == "dev":
            return self.args.dev_path
        if self.corpus_type == "test":
            return self.args.test_path
        return None

    def parse_tag_vocab(self):
        with open(self.args.tag_vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.tag_vocab.append(line)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        token_text = [tok.text for tok in sentence.tokens]
        tag_ids = [self.get_tag_index(tok.tag) for tok in sentence.tokens]

        return {"text": token_text,
                "labels": tag_ids}

    def get_tag_index(self, text_tag):
        if text_tag not in self.tag_vocab:
            text_tag = self.args.none_tag
        return self.tag_vocab.index(text_tag)


@dataclass
class NerCharDataCollator:
    args: AdditionalArguments

    def __call__(self, features):
        # post-padding
        max_len = max(len(entry["labels"]) for entry in features)
        batch = dict()

        # input_ids
        batch_text = [entry["text"] for entry in features]
        char_vocab = NerDataset.get_char_vocab()
        batch["input_ids"] = NerDataset.get_char_ids(batch_text, max_len, char_vocab)

        # attention_mask
        entry = []
        for i in range(len(features)):
            good_len = len(features[i]["labels"])
            pad_len = max_len - good_len
            entry.append(torch.tensor([1] * good_len + [0] * pad_len))
        batch["attention_mask"] = torch.stack(entry)

        # labels
        entry = []
        for i in range(len(features)):
            pad_len = max_len - len(features[i]["labels"])
            entry.append(torch.tensor(features[i]["labels"] + [-100] * pad_len))
        batch["labels"] = torch.stack(entry)

        return batch


def main(args):
    setup_logging()
    parser = HfArgumentParser([AdditionalArguments])
    additional_args = parse_config(parser, args.config)[0]
    set_all_seeds(42)
    dataset = NerCharDataset(additional_args, corpus_type="test")
    print(len(dataset))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Char Dataset Runner")
    ap.add_argument("--config", default="config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
