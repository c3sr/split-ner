import argparse
import logging
import os

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction

from secner.dataset import NerDataset
from secner.utils.general import Sentence, Token, setup_logging

logger = logging.getLogger(__name__)


class PreviousSentenceSelector:
    def __init__(self, args):
        self.args = args
        self.train_path = os.path.join("..", "data", args.input_dataset, "train.tsv")
        self.dataset_path = os.path.join("..", "data", args.input_dataset, args.file)
        self.output_dataset_path = os.path.join("..", "data", args.output_dataset, args.file)
        training_sentences = NerDataset.read_dataset(self.train_path, self.args)
        self.candidates = [Sentence([Token(text=token.text,
                                           tags=[args.none_tag],
                                           pos_tag=token.pos_tag,
                                           dep_tag=token.dep_tag,
                                           guidance_tag=token.guidance_tag)
                                     for token in sent.tokens]) for sent in training_sentences]
        self.dataset = NerDataset.read_dataset(self.dataset_path, self.args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        self.model = AutoModelForNextSentencePrediction.from_pretrained(args.base_model).to(args.device)
        self.model.eval()

    def get_best_prev_sentence(self, sentence):
        text = " ".join(token.text for token in sentence.tokens)
        sz = self.args.batch_size
        text_candidates = []
        outputs = []
        for i in tqdm(range(0, len(self.candidates), sz)):
            candidates = [" ".join([tok.text for tok in sent.tokens]) for sent in self.candidates[i: i + sz]]
            text_sent = [text] * len(candidates)
            encodings = self.tokenizer(candidates, text_sent, return_tensors="pt", padding=True).to(self.args.device)
            logits = self.model(**encodings).logits
            outputs.extend(logits.detach().cpu().numpy()[:, 0].tolist())
            text_candidates.extend(candidates)
        outputs = np.array(outputs)
        if text in text_candidates:
            pos = text_candidates.index(text)
            outputs[pos] = -float("inf")
        index = int(np.argmax(outputs))
        new_sentence = Sentence(self.candidates[index].tokens + sentence.tokens)
        return new_sentence

    def make_new_dataset(self):
        open(self.output_dataset_path, "w", encoding="utf-8").close()
        for index, sent in enumerate(self.dataset):
            if index < self.args.resume:
                continue
            logger.info("processing dataset sentence: {0}".format(index))
            new_sent = self.get_best_prev_sentence(sent)
            with open(self.output_dataset_path, "a", encoding="utf-8") as f:
                f.write("{0}\n\n".format(new_sent.to_tsv_form()))


def main(args):
    setup_logging()
    selector = PreviousSentenceSelector(args)
    selector.make_new_dataset()


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Previous Sentence Selector")
    ap.add_argument("--input_dataset", type=str, default="bio", help="input dataset dir name")
    ap.add_argument("--output_dataset", type=str, default="bio_pair", help="output paired dataset dir name")
    ap.add_argument("--file", type=str, default="train.tsv", help="file (train.tsv/dev.tsv/test.tsv) to convert")
    ap.add_argument("--base_model", type=str, default="dmis-lab/biobert-base-cased-v1.1", help="base model name")
    ap.add_argument("--batch_size", type=int, default=16, help="batch size")
    ap.add_argument("--resume", type=int, default=0, help="resume from sentence index (default: 0)")
    ap.add_argument("--device", type=str, default="cuda:0", help="device")
    ap.add_argument("--none_tag", type=str, default="O", help="none tag")
    ap = ap.parse_args()
    main(ap)
