import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from secner.base import BaseExecutor
from secner.dataset import NerDataset
from secner.evaluator import Evaluator
from secner.model import NerModel
from secner.utils import set_all_seeds, parse_config


class NerExecutor(BaseExecutor):

    def __init__(self, config):
        super(NerExecutor, self).__init__(config)

        self.train_dataset = NerDataset(self.config, "train")
        self.dev_dataset = NerDataset(self.config, "dev")
        self.test_dataset = NerDataset(self.config, "test")

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=False,
                                            collate_fn=self.train_dataset.collate)
        self.dev_data_loader = DataLoader(self.dev_dataset, batch_size=self.config.batch_size, shuffle=False,
                                          collate_fn=self.dev_dataset.collate)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                           collate_fn=self.test_dataset.collate)

        self.num_tags = 34
        bert_config = BertConfig.from_pretrained("bert-base-uncased", num_labels=self.num_tags)
        self.model = NerModel.from_pretrained("bert-base-uncased", config=bert_config, ner_params=self.config)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=self.config.lr)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = []
        with tqdm(self.train_data_loader) as progress_bar:
            for text, attention_mask, token_ids, offsets, tag_ids in progress_bar:
                self.optimizer.zero_grad()
                loss = self.model(token_ids, attention_mask, tag_ids)
                progress_bar.set_postfix(Epoch=epoch, Batch_Loss="{0:.3f}".format(loss.item()))
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
        print("TRAIN: Epoch: {0} | Loss:{1:.3f}".format(epoch, sum(train_loss) / len(train_loss)))

    def evaluate_epoch(self, data_loader, epoch, prefix, outfile=None):
        self.model.eval()
        total_text = []
        total_offsets = []
        total_prediction = []
        total_tags = []
        with tqdm(data_loader) as progress_bar:
            for text, attention_mask, token_ids, offsets, tag_ids in progress_bar:
                total_text.extend(text)
                total_offsets.extend(offsets.detach().cpu().numpy().tolist())
                with torch.no_grad():
                    prediction_ids = self.model(token_ids, attention_mask)
                    total_prediction.extend(prediction_ids.detach().cpu().numpy().tolist())
                    total_tags.extend(tag_ids.detach().cpu().numpy().tolist())
        if outfile:
            self.print_outputs(corpus=total_text, gold=total_tags, predicted=total_prediction, offsets=total_offsets,
                               mapping=data_loader.dataset.tag_vocab, outfile=outfile)
        evaluator = Evaluator(gold=total_tags, predicted=total_prediction, tags=data_loader.dataset.tag_vocab)
        print("Entity-Level Metrics:")
        print(evaluator.entity_metric.report())

        return evaluator


def main(args):
    config = parse_config(args.config)
    set_all_seeds(config.seed)
    executor = NerExecutor(config)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Model Runner")
    ap.add_argument("--config", default="config.json", help="config json file (Default: config.json)")
    ap = ap.parse_args()
    main(ap)
