import argparse
import logging
import numpy as np
import os
from transformers import AutoConfig
from transformers import HfArgumentParser
from transformers.trainer import TrainingArguments

from secner.additional_args import AdditionalArguments
from secner.dataset_char import NerCharDataset, NerCharDataCollator
from secner.evaluator import Evaluator
from secner.model_char import NerModelChar
from secner.trainer import NerTrainer
from secner.utils.general import set_all_seeds, set_wandb, parse_config, setup_logging

logger = logging.getLogger(__name__)


class NerCharExecutor:
    def __init__(self, train_args: TrainingArguments, additional_args: AdditionalArguments):
        os.environ["WANDB_MODE"] = additional_args.wandb_mode
        set_wandb(additional_args.wandb_dir)
        logger.info("training args: {0}".format(train_args.to_json_string()))
        logger.info("additional args: {0}".format(additional_args.to_json_string()))
        set_all_seeds(train_args.seed)

        self.train_args = train_args
        self.additional_args = additional_args

        self.train_dataset = NerCharDataset(additional_args, "train")
        self.dev_dataset = NerCharDataset(additional_args, "dev")
        self.test_dataset = NerCharDataset(additional_args, "test")

        self.num_labels = len(self.train_dataset.tag_vocab)
        model_path = additional_args.resume if additional_args.resume else additional_args.base_model
        bert_config = AutoConfig.from_pretrained(model_path, num_labels=self.num_labels)

        self.model = NerModelChar.from_pretrained(model_path, config=bert_config, additional_args=additional_args)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        logger.info("# trainable params: {0}".format(sum([np.prod(p.size()) for p in trainable_params])))

        data_collator = NerCharDataCollator(args=self.additional_args)
        self.trainer = NerTrainer(model=self.model,
                                  args=train_args,
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
        data = self.get_token_mapping(dataset, model_predictions)

        os.makedirs(self.additional_args.predictions_dir, exist_ok=True)
        predictions_file = os.path.join(self.additional_args.predictions_dir, "{0}.tsv".format(dataset.corpus_type))
        logger.info("Outputs published in file: {0}".format(predictions_file))
        with open(predictions_file, "w", encoding="utf-8") as f:
            # f.write("Token\tGold\tPredicted\n")
            for sent in data:
                for word in sent:
                    f.write("{0}\t{1}\t{2}\n".format(word[0], word[1], word[2]))
                f.write("\n")

    def get_token_mapping(self, dataset, model_predictions):
        data = []
        for i in range(len(dataset)):
            sentence = dataset.sentences[i]
            prediction = model_predictions[i]
            data.append([[tok.text, tok.tag, dataset.tag_vocab[prediction[j]]]
                         for j, tok in enumerate(sentence.tokens)])
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
    executor = NerCharExecutor(train_args, additional_args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Char CNN Model Runner")
    ap.add_argument("--config", default="config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
