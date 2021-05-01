import argparse
import logging
import os

import numpy as np
from transformers import AutoConfig, AutoTokenizer
from transformers import HfArgumentParser
from transformers.trainer import TrainingArguments

from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataset, NerDataCollator
from secner.dataset_char import NerCharDataset, NerCharDataCollator
from secner.evaluator import Evaluator
from secner.model import NerModel
from secner.model_bidaf import NerModelBiDAF
from secner.model_char import NerModelChar
from secner.model_crf import NerModelWithCrf
from secner.model_roberta import NerRobertaModel
from secner.trainer import NerTrainer
from secner.utils.general import set_all_seeds, set_wandb, parse_config, setup_logging

logger = logging.getLogger(__name__)


class NerExecutor:
    def __init__(self, train_args: TrainingArguments, additional_args: AdditionalArguments):
        os.environ["WANDB_MODE"] = additional_args.wandb_mode
        set_wandb(additional_args.wandb_dir)
        logger.info("training args: {0}".format(train_args.to_json_string()))
        logger.info("additional args: {0}".format(additional_args.to_json_string()))
        set_all_seeds(train_args.seed)

        self.train_args = train_args
        self.additional_args = additional_args

        dataset_class = self.get_dataset_class()
        self.train_dataset = dataset_class(additional_args, "train")
        self.dev_dataset = dataset_class(additional_args, "dev")
        self.test_dataset = dataset_class(additional_args, "test")

        self.num_labels = len(self.train_dataset.tag_vocab)
        model_path = additional_args.resume if additional_args.resume else additional_args.base_model
        bert_config = AutoConfig.from_pretrained(model_path, num_labels=self.num_labels)

        model_class = self.get_model_class()
        # load best model in end fails as additional_args is not passed in Trainer (but training successfully completes)
        self.model = model_class.from_pretrained(model_path, config=bert_config, additional_args=additional_args)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        logger.info("# trainable params: {0}".format(sum([np.prod(p.size()) for p in trainable_params])))

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.trainer = NerTrainer(model=self.model,
                                  args=train_args,
                                  tokenizer=self.tokenizer,
                                  data_collator=self.get_data_collator(),
                                  train_dataset=self.train_dataset,
                                  eval_dataset=self.dev_dataset,
                                  compute_metrics=self.compute_metrics)

    def compute_metrics(self, eval_prediction):
        evaluator = Evaluator(gold=eval_prediction.label_ids, predicted=eval_prediction.predictions,
                              tags=self.dev_dataset.tag_vocab, tagging_scheme=self.additional_args.tagging)
        logger.info("entity metrics:\n{0}".format(evaluator.entity_metric.report()))
        return {"micro_f1": evaluator.entity_metric.micro_avg_f1()}

    def dump_predictions(self, dataset):
        model_predictions: np.ndarray = self.trainer.predict(dataset).predictions
        data = self.bert_to_orig_token_mapping1(dataset, model_predictions)
        # data = self.bert_to_orig_token_mapping2(dataset, model_predictions)

        os.makedirs(self.additional_args.predictions_dir, exist_ok=True)
        predictions_file = os.path.join(self.additional_args.predictions_dir, "{0}.tsv".format(dataset.corpus_type))
        logger.info("Outputs published in file: {0}".format(predictions_file))
        with open(predictions_file, "w", encoding="utf-8") as f:
            # f.write("Token\tGold\tPredicted\n")
            for sent in data:
                for word in sent:
                    f.write("{0}\t{1}\t{2}\n".format(word[0], word[1], word[2]))
                f.write("\n")

    # take the tag output for the first bert token as the tag for the original token
    # slightly more: "true positives", slightly less: "false positives", "false negatives"
    def bert_to_orig_token_mapping1(self, dataset, model_predictions):
        data = []
        pad_tag = self.additional_args.pad_tag
        for i in range(len(dataset)):
            sentence = dataset.sentences[i]
            prediction = model_predictions[i]
            # considering only the first gold tag associated with the token
            data.append([[tok.text, tok.tags[0], pad_tag] for tok in sentence.tokens])
            if self.additional_args.use_head_mask:
                for j in range(len(sentence.tokens)):
                    data[i][j][2] = dataset.tag_vocab[prediction[j + 1]]
            else:
                offsets = [tok.token.offset for tok in sentence.bert_tokens]
                ptr = 0
                r = min(prediction.shape[0], len(offsets))
                for j in range(r):
                    if offsets[j] != ptr:
                        continue
                    data[i][ptr][2] = dataset.tag_vocab[prediction[j]]
                    ptr += 1
            for j in range(len(data[i])):
                if data[i][j][2].startswith("S-"):
                    data[i][j][2] = "B-" + data[i][j][2][2:]
                elif data[i][j][2].startswith("E-"):
                    data[i][j][2] = "I-" + data[i][j][2][2:]
        return data

    # for each original token, if the output for bert sub-tokens is inconsistent, then map to NONE_TAG else take the tag
    # slightly more: "true positives", slightly less: "false negatives", considerably less: "false positives"
    def bert_to_orig_token_mapping2(self, dataset, model_predictions):
        data = []
        pad_tag = self.additional_args.pad_tag
        none_tag = self.additional_args.none_tag
        for i in range(len(dataset)):
            sentence = dataset.sentences[i]
            prediction = model_predictions[i]
            # considering only the first gold tag associated with the token
            data.append([[tok.text, tok.tags[0], pad_tag] for tok in sentence.tokens])
            offsets = [tok.token.offset for tok in sentence.bert_tokens]
            ptr = -1
            r = min(prediction.shape[0], len(offsets))
            for j in range(1, r - 1):
                if offsets[j] > ptr:
                    ptr += 1
                    data[i][ptr][2] = dataset.tag_vocab[prediction[j]]
                elif ("I-" + data[i][ptr][2][2:]) != dataset.tag_vocab[prediction[j]]:
                    data[i][ptr][2] = none_tag
        return data

    def run(self):
        if self.train_args.do_train:
            logger.info("training mode")
            self.trainer.train(self.additional_args.resume)
        else:
            logger.info("prediction mode")
            assert self.additional_args.resume is not None, "specify model checkpoint to load for predictions"
            self.dump_predictions(self.train_dataset)
            self.dump_predictions(self.dev_dataset)
            self.dump_predictions(self.test_dataset)
            # throws some threading related tqdm/wandb exception in the end (but code fully works)

    def get_model_class(self):
        if self.additional_args.model_mode == "std":
            return NerModel
        if self.additional_args.model_mode == "roberta_std":
            return NerRobertaModel
        if self.additional_args.model_mode == "crf":
            return NerModelWithCrf
        if self.additional_args.model_mode == "bidaf":
            return NerModelBiDAF
        if self.additional_args.model_mode == "char":
            return NerModelChar

    def get_dataset_class(self):
        if self.additional_args.model_mode == "char":
            return NerCharDataset
        return NerDataset

    def get_data_collator(self):
        if self.additional_args.model_mode == "char":
            return NerCharDataCollator(args=self.additional_args)

        # return DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        return NerDataCollator(args=self.additional_args)


def main(args):
    setup_logging()
    parser = HfArgumentParser([TrainingArguments, AdditionalArguments])
    train_args, additional_args = parse_config(parser, args.config)
    executor = NerExecutor(train_args, additional_args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Model Runner")
    ap.add_argument("--config", default="config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
