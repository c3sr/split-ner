import argparse
import logging
import os

import numpy as np
from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataset, NerDataCollator
from secner.evaluator import Evaluator
from secner.model_crf import NerModelWithCrf
from secner.trainer import NerTrainer
from secner.utils.general import set_all_seeds, set_wandb, parse_config, setup_logging
from transformers import AutoConfig, AutoTokenizer
from transformers import HfArgumentParser
from transformers.trainer import TrainingArguments

logger = logging.getLogger(__name__)


class NerExecutorCrf:
    def __init__(self, train_args, additional_args):
        set_wandb(additional_args.wandb_dir)
        logger.info("training args: {0}".format(train_args.to_json_string()))
        logger.info("additional args: {0}".format(additional_args.to_json_string()))
        set_all_seeds(train_args.seed)

        self.train_args = train_args
        self.additional_args = additional_args

        self.train_dataset = NerDataset(additional_args, "train")
        self.dev_dataset = NerDataset(additional_args, "dev")
        self.test_dataset = NerDataset(additional_args, "test")

        self.num_labels = additional_args.num_labels
        model_path = additional_args.resume if additional_args.resume else additional_args.base_model
        bert_config = AutoConfig.from_pretrained(model_path, num_labels=self.num_labels)
        self.model = NerModelWithCrf.from_pretrained(model_path, config=bert_config, additional_args=additional_args)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        logger.info("# trainable params: {0}".format(sum([np.prod(p.size()) for p in trainable_params])))

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        data_collator = NerDataCollator(args=additional_args)
        self.trainer = NerTrainer(model=self.model,
                                  args=train_args,
                                  tokenizer=tokenizer,
                                  data_collator=data_collator,
                                  train_dataset=self.train_dataset,
                                  eval_dataset=self.dev_dataset,
                                  compute_metrics=self.compute_metrics)

    def compute_metrics(self, eval_prediction):
        evaluator = Evaluator(gold=eval_prediction.label_ids, predicted=eval_prediction.predictions,
                              tags=self.dev_dataset.tag_vocab)
        logger.info("entity metrics:\n{0}".format(evaluator.entity_metric.report()))
        return {"micro_f1": evaluator.entity_metric.micro_avg_f1()}

    def dump_predictions(self, dataset):
        model_predictions: np.ndarray = self.trainer.predict(dataset).predictions
        data = self.bert_to_orig_token_mapping1(dataset, model_predictions)
        # data = self.bert_to_orig_token_mapping2(dataset, model_predictions)

        os.makedirs(self.additional_args.predictions_dir, exist_ok=True)
        predictions_file = os.path.join(self.additional_args.predictions_dir, "{0}.tsv".format(dataset.corpus_type))
        logger.info("Outputs published in file: {0}".format(predictions_file))
        with open(predictions_file, "w") as f:
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
            data.append([[tok.text, tok.tag, pad_tag] for tok in sentence.tokens])
            offsets = [tok.token.offset for tok in sentence.bert_tokens]
            ptr = 0
            r = min(prediction.shape[0], len(offsets))
            for j in range(r):
                if offsets[j] != ptr:
                    continue
                data[i][ptr][2] = dataset.tag_vocab[prediction[j]]
                ptr += 1
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
            data.append([[tok.text, tok.tag, pad_tag] for tok in sentence.tokens])
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


def main(args):
    setup_logging()
    parser = HfArgumentParser([TrainingArguments, AdditionalArguments])
    train_args, additional_args = parse_config(parser, args.config)
    executor = NerExecutorCrf(train_args, additional_args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CRF-based Model Runner")
    ap.add_argument("--config", default="config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
