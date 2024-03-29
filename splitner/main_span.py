import argparse
import logging
import os
import time
from datetime import datetime, timedelta
import traceback

import numpy as np
from transformers import AutoConfig, AutoTokenizer
from transformers import HfArgumentParser
from transformers.trainer import TrainingArguments

from splitner.additional_args import AdditionalArguments
from splitner.dataset_span import NerSpanDataCollator, NerSpanDataset
from splitner.evaluator_span import EvaluatorSpan
from splitner.trainer import NerTrainer
from splitner.utils.general import set_all_seeds, set_wandb, parse_config, setup_logging

logger = logging.getLogger(__name__)


class NerSpanExecutor:
    # TODO: Currently this model is trained on valid correct spans as input and does not output NONE tag for any input
    def __init__(self, train_args: TrainingArguments, additional_args: AdditionalArguments):
        os.environ["WANDB_MODE"] = additional_args.wandb_mode
        set_wandb(additional_args.wandb_dir)
        logger.info("training args: {0}".format(train_args.to_json_string()))
        logger.info("additional args: {0}".format(additional_args.to_json_string()))
        set_all_seeds(train_args.seed)

        self.train_args = train_args
        self.additional_args = additional_args

        self.train_dataset = NerSpanDataset(additional_args, "train")
        self.dev_dataset = NerSpanDataset(additional_args, "dev")
        self.test_dataset = NerSpanDataset(additional_args, "test")

        self.num_labels = len(self.train_dataset.tag_vocab)

        model_path = additional_args.resume if additional_args.resume else additional_args.base_model
        bert_config = AutoConfig.from_pretrained(model_path, num_labels=self.num_labels)

        model_class = self.get_model_class()
        self.model = model_class.from_pretrained(model_path, config=bert_config, additional_args=additional_args)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        logger.info("# trainable params: {0}".format(sum([np.prod(p.size()) for p in trainable_params])))

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        data_collator = NerSpanDataCollator(args=additional_args)
        self.trainer = NerTrainer(model=self.model,
                                  args=train_args,
                                  tokenizer=tokenizer,
                                  data_collator=data_collator,
                                  train_dataset=self.train_dataset,
                                  eval_dataset=self.dev_dataset,
                                  compute_metrics=self.compute_metrics)

    def compute_metrics(self, eval_prediction):
        evaluator = EvaluatorSpan(gold=eval_prediction.label_ids, predicted=eval_prediction.predictions,
                                  tags=self.dev_dataset.tag_vocab)
        logger.info("entity metrics:\n{0}".format(evaluator.entity_metric.report()))
        return {"micro_f1": evaluator.entity_metric.micro_avg_f1()}

    def dump_predictions(self, dataset):
        os.makedirs(self.additional_args.predictions_dir, exist_ok=True)
        timer_file_path = os.path.join(self.additional_args.predictions_dir, "{0}-timer.log".format(dataset.corpus_type))
        timer_file = open(timer_file_path, "a")
        
        total_elapsed = 0
        n = 1   # manually set to 10 for prod experiments
        for i in range(0, n):
            logger.info("{0}-th prediction".format(str(i)))
            start = time.time()

            model_predictions: np.ndarray = self.trainer.predict(dataset).predictions
            data = self.map_predictions_to_sentences(dataset, model_predictions)

            elapsed = time.time() - start
            logger.info("elapsed time: {0} seconds: {1}".format(str(elapsed), str(timedelta(seconds=elapsed))))

            total_elapsed += elapsed
            timer_file.write(f"Iteration {str(i)}: {str(elapsed)}\n")

        avg_elapsed = total_elapsed / n
        timer_file.write(f"Avg: {str(avg_elapsed)}\n")
        timer_file.close()

        predictions_file = os.path.join(self.additional_args.predictions_dir, "{0}.tsv".format(dataset.corpus_type))
        logger.info("Outputs published in file: {0}".format(predictions_file))
        with open(predictions_file, "w", encoding="utf-8") as f:
            # f.write("Token\tGold\tPredicted\n")
            for sent in data:
                for word in sent:
                    f.write("{0}\t{1}\t{2}\n".format(word[0], word[1], word[2]))
                f.write("\n")

    def map_predictions_to_sentences(self, dataset, model_predictions):
        data_dict = {}
        none_tag = self.additional_args.none_tag
        for i in range(len(dataset)):
            context = dataset.contexts[i]
            text_sentence = " ".join([tok.text for tok in context.sentence.tokens])
            predicted_entity = self.dev_dataset.tag_vocab[model_predictions[i]]
            if text_sentence not in data_dict:
                # considering only the first gold tag associated with the token
                data_dict[text_sentence] = [[tok.text, tok.tags[0], none_tag] for tok in context.sentence.tokens]
            data_dict[text_sentence][context.mention_span.start][2] = "B-{0}".format(predicted_entity)
            for index in range(context.mention_span.start + 1, context.mention_span.end + 1):
                data_dict[text_sentence][index][2] = "I-{0}".format(predicted_entity)

        data = []
        for context in dataset.contexts:
            text_sentence = " ".join([tok.text for tok in context.sentence.tokens])
            if text_sentence in data_dict:
                data.append(data_dict[text_sentence])
                del data_dict[text_sentence]

        return data

    def get_model_class(self):
        if self.additional_args.model_mode == "std":
            from splitner.model_span import NerSpanModel
            return NerSpanModel
        if self.additional_args.model_mode == "roberta_std":
            from splitner.model_span_roberta import NerSpanRobertaModel
            return NerSpanRobertaModel
        raise NotImplementedError

    def run(self):
        if self.train_args.do_train:
            start = time.time()
            logger.info("training mode: start time: {0}".format(str(datetime.now())))

            try:
                self.trainer.train(self.additional_args.resume)
            except:
                traceback.print_exc()

            end = time.time()
            logger.info("end time: {0}".format(str(datetime.now())))
            elapsed = end - start
            logger.info("Elapsed time: {0} | In seconds: {1}".format(str(elapsed), str(timedelta(seconds=elapsed))))
        else:
            if self.additional_args.infer_inp_path:
                logger.info("inference mode")
                from splitner.dataset_span import NerInferSpanDataset
                self.dump_predictions(NerInferSpanDataset(self.additional_args))
            else:
                logger.info("prediction mode")
                # self.dump_predictions(self.train_dataset)
                # self.dump_predictions(self.dev_dataset)
                self.dump_predictions(self.test_dataset)
                # throws some threading related tqdm/wandb exception in the end (but code fully works)


def main():
    setup_logging()
    parser = HfArgumentParser([TrainingArguments, AdditionalArguments])

    import sys
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # when a config json file is provided, parse it to get our arguments.
        train_args, additional_args = parse_config(parser, sys.argv[1])
    else:
        train_args, additional_args = parser.parse_args_into_dataclasses()

    executor = NerSpanExecutor(train_args, additional_args)
    executor.run()

if __name__ == "__main__":
    main()
