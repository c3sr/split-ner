import argparse
import logging
import os

import numpy as np
from transformers import AutoConfig, AutoTokenizer
from transformers import HfArgumentParser
from transformers.trainer import TrainingArguments

from secner.additional_args import AdditionalArguments
from secner.dataset import NerDataCollator
from secner.dataset_qa import NerQADataset
from secner.evaluator_qa import EvaluatorQA
from secner.model_bidaf import NerModelBiDAF
from secner.trainer import NerTrainer
from secner.utils.general import set_all_seeds, set_wandb, parse_config, setup_logging

logger = logging.getLogger(__name__)


class NerQAExecutor:
    def __init__(self, train_args, additional_args):
        set_wandb(additional_args.wandb_dir)
        logger.info("training args: {0}".format(train_args.to_json_string()))
        logger.info("additional args: {0}".format(additional_args.to_json_string()))
        set_all_seeds(train_args.seed)

        self.train_args = train_args
        self.additional_args = additional_args

        self.train_dataset = NerQADataset(additional_args, "train")
        self.dev_dataset = NerQADataset(additional_args, "dev")
        self.test_dataset = NerQADataset(additional_args, "test")

        # num_labels = 3 (for BIO tagging scheme), num_labels = 4 (for BIOE tagging scheme)
        self.num_labels = self.additional_args.num_labels

        model_path = additional_args.resume if additional_args.resume else additional_args.base_model
        bert_config = AutoConfig.from_pretrained(model_path, num_labels=self.num_labels)
        self.model = NerModelBiDAF.from_pretrained(model_path, config=bert_config, additional_args=additional_args)

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
        predictions = np.argmax(eval_prediction.predictions, axis=2)
        evaluator = EvaluatorQA(gold=eval_prediction.label_ids, predicted=predictions, num_labels=self.num_labels,
                                none_tag=self.additional_args.none_tag)
        logger.info("entity metrics:\n{0}".format(evaluator.entity_metric.report()))
        return {"micro_f1": evaluator.entity_metric.micro_avg_f1()}

    def dump_predictions(self, dataset: NerQADataset):
        model_predictions: np.ndarray = np.argmax(self.trainer.predict(dataset).predictions, axis=2)
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
        data_dict = {}
        pad_tag = self.additional_args.pad_tag
        none_tag = self.additional_args.none_tag
        for i in range(len(dataset)):
            context = dataset.contexts[i]
            text_sentence = " ".join([tok.text for tok in context.sentence.tokens])
            prediction = model_predictions[i]
            if text_sentence not in data_dict:
                data_dict[text_sentence] = [[tok.text, tok.tag, pad_tag] for tok in context.sentence.tokens]
            ptr = 0
            r = min(prediction.shape[0], len(context.bert_tokens))
            for j in range(r):
                if context.bert_tokens[j].token_type == 0:
                    continue

                if context.bert_tokens[j].token.offset != ptr:
                    continue

                if data_dict[text_sentence][ptr][2] not in [pad_tag, none_tag]:
                    ptr += 1
                    continue

                if prediction[j] == NerQADataset.get_tag_index("B", none_tag):
                    tag_assignment = "B-" + context.entity
                elif prediction[j] == NerQADataset.get_tag_index("I", none_tag) or \
                        prediction[j] == NerQADataset.get_tag_index("E", none_tag):
                    tag_assignment = "I-" + context.entity
                else:
                    tag_assignment = none_tag

                data_dict[text_sentence][ptr][2] = tag_assignment
                ptr += 1

        data = []
        for context in dataset.contexts:
            text_sentence = " ".join([tok.text for tok in context.sentence.tokens])
            if text_sentence in data_dict:
                data.append(data_dict[text_sentence])
                del data_dict[text_sentence]

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
    executor = NerQAExecutor(train_args, additional_args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QA BiDAF Model Runner")
    ap.add_argument("--config", default="config/bio/config_biobert_qa4_bidaf.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
