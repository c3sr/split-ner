import os
import re

import numpy as np
import torch
from tqdm import tqdm

from src.utils.evaluator import Evaluator
from src.utils.general import get_compute_device


class BaseExecutor:
    CHECKPOINT_PATTERN = re.compile("checkpoint-epoch-(\\d+)\\.pth")

    def __init__(self, config):
        self.config = config
        self.device = get_compute_device(self.config.use_cpu)
        print("using device: {0}".format(self.device))

        self.checkpoint_dir = os.path.join(self.config.checkpoint_dir, self.config.name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.config.data.out_dir = os.path.join(self.config.data.out_dir, self.config.name)
        if not os.path.exists(self.config.data.out_dir):
            os.makedirs(self.config.data.out_dir)

        self.config.data.train_path = os.path.join(self.config.data.data_dir, self.config.data.train_path)
        self.config.data.dev_path = os.path.join(self.config.data.data_dir, self.config.data.dev_path)
        self.config.data.test_path = os.path.join(self.config.data.data_dir, self.config.data.test_path)
        self.config.verbose = not self.config.no_eval_print

        self.config.pad_tag = "<PAD>"
        self.config.unk_tag = "<UNK>"
        self.config.none_tag = "O"

        self.shuffle_train_data = not (self.config.eval != "none" or self.config.query)

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        self.train_data_loader = None
        self.dev_data_loader = None
        self.test_data_loader = None

        self.model = None
        self.criterion = None
        self.optimizer = None

        self.start_epoch = 0
        self.monitor_best = 0.0

    def print_model_summary(self):
        print("Run Configuration:")
        print("------------------")
        for arg in vars(self.config):
            print("{0}: {1}".format(arg, getattr(self.config, arg)))
        print()

        print("Model Summary:")
        print("--------------")
        print(self.model)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        num_params = sum([np.prod(p.size()) for p in trainable_params])
        print("No. of trainable parameters: {0}".format(num_params))

    def run(self):
        self.resume_checkpoint()
        self.print_model_summary()

        if self.config.query:
            self.query(input("Query Sentence:"))
        elif self.config.eval != "none":
            return self.evaluate()
        else:
            self.train()

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.config.num_epochs):
            self.train_epoch(epoch)
            self.save_checkpoint(epoch, save_best=False)

            _, dev_evaluator = self.evaluate_epoch(self.dev_data_loader, epoch=epoch, prefix="DEV")
            _, test_evaluator = self.evaluate_epoch(self.test_data_loader, epoch=epoch, prefix="TEST")

            dev_micro_f1 = dev_evaluator.entity_metric.micro_avg_f1()
            is_curr_best = dev_micro_f1 >= self.monitor_best
            if is_curr_best:
                print("Found new best model with dev entity-level Micro F1: {0:.3f}. (prev. best dev Micro F1: {1:.3f})"
                      .format(dev_micro_f1, self.monitor_best))
                self.monitor_best = dev_micro_f1
            self.save_checkpoint(epoch, save_best=is_curr_best)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_prediction = []
        train_label = []
        with tqdm(self.train_data_loader) as progress_bar:
            for text, feature, label in progress_bar:
                feature = feature.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                prediction = self.model(feature)
                train_prediction.append(prediction.detach().clone())
                train_label.append(label.detach().clone())
                prediction = prediction.transpose(2, 1)
                loss = self.criterion(prediction, label)
                progress_bar.set_postfix(Epoch=epoch, Batch_Loss="{0:.3f}".format(loss.item()))
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

        train_prediction = torch.cat(train_prediction, dim=0).cpu().numpy()
        train_prediction = np.argmax(train_prediction, axis=2)
        train_label = torch.cat(train_label, dim=0).cpu().numpy()
        evaluator = Evaluator(gold=train_label, predicted=train_prediction, tags=self.train_dataset.out_tags,
                              ignore_tags=[self.config.none_tag, self.config.pad_tag], none_tag=self.config.none_tag,
                              pad_tag=self.config.pad_tag)
        print("TRAIN: Epoch: {0} | Loss:{1:.3f} | Token-Level Micro F1: {2:.3f}".format(epoch, train_loss / len(
            self.train_data_loader.dataset), evaluator.significant_token_metric.micro_avg_f1()))

    def evaluate(self):
        if not os.path.exists(self.config.out_dir):
            os.makedirs(self.config.out_dir)
        train_outfile = os.path.join(self.config.out_dir, "train.out.tsv")
        dev_outfile = os.path.join(self.config.out_dir, "dev.out.tsv")
        test_outfile = os.path.join(self.config.out_dir, "test.out.tsv")
        last_trained_epoch = self.start_epoch - 1
        _, train_evaluator = self.evaluate_epoch(self.train_data_loader, epoch=last_trained_epoch, prefix="TRAIN",
                                                 outfile=train_outfile)
        _, dev_evaluator = self.evaluate_epoch(self.dev_data_loader, epoch=last_trained_epoch, prefix="DEV",
                                               outfile=dev_outfile)
        _, test_evaluator = self.evaluate_epoch(self.test_data_loader, epoch=last_trained_epoch, prefix="TEST",
                                                outfile=test_outfile)

        return train_evaluator, dev_evaluator, test_evaluator

    def evaluate_epoch(self, data_loader, epoch, prefix, outfile=None):
        self.model.eval()
        total_loss = 0.0
        total_text = []
        total_prediction = []
        total_label = []
        for text, feature, label in data_loader:
            text = np.array(text).T.tolist()
            feature = feature.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                prediction = self.model(feature)
                total_prediction.append(prediction.clone())
                total_label.append(label.clone())
                total_text.extend(text)
                prediction = prediction.transpose(2, 1)
                loss = self.criterion(prediction, label)
                total_loss += loss.item()
        total_prediction = torch.cat(total_prediction, dim=0).cpu().numpy()
        total_prediction = np.argmax(total_prediction, axis=2)
        total_label = torch.cat(total_label, dim=0).cpu().numpy()
        if outfile:
            self.print_outputs(corpus=total_text, gold=total_label, predicted=total_prediction,
                               mapping=data_loader.dataset.out_tags, outfile=outfile)
        evaluator = Evaluator(gold=total_label, predicted=total_prediction, tags=data_loader.dataset.out_tags,
                              ignore_tags=[self.config.none_tag, self.config.pad_tag], none_tag=self.config.none_tag,
                              pad_tag=self.config.pad_tag)
        mean_loss = total_loss / len(data_loader.dataset)
        print("{0}: Epoch: {1} | Token-Level Micro F1: {2:.3f} | Loss: {3:.3f}".format(
            prefix, epoch, evaluator.significant_token_metric.micro_avg_f1(), mean_loss))
        if self.config.verbose:
            print("Entity-Level Metrics:")
            print(evaluator.entity_metric.report())
            print("Token-Level (Without 'O' Tag) Metrics:")
            print(evaluator.significant_token_metric.report())

        return mean_loss, evaluator

    def query(self, sentence_text):
        self.model.eval()

        sentence_tokens = BaseExecutor.get_query_tokens(sentence_text)
        text, feature, _ = self.test_dataset.get_query_given_tokens(sentence_tokens)
        feature = torch.as_tensor(feature, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(feature)
        prediction = np.argmax(prediction, axis=2).squeeze(0)
        for i in range(prediction.shape[0]):
            print("{0}\t{1}".format(text[i], self.test_dataset.out_tags[prediction[i]]))

    @staticmethod
    def get_query_tokens(sentence_text):
        from spacy.lang.en import English
        nlp = English()
        tokenizer = nlp.Defaults.create_tokenizer(nlp)
        return [tok.text for tok in tokenizer(sentence_text)]

    def print_outputs(self, corpus, gold, predicted, mapping, outfile, mask=None):
        print("Outputs published in file: {0}".format(outfile))
        if not isinstance(mask, np.ndarray):
            mask = np.ones(gold.shape)
        with open(outfile, "w") as f:
            f.write("Token\tGold\tPredicted\n")
            for sent_index in range(len(corpus)):
                for word_index in range(len(corpus[sent_index])):
                    token = corpus[sent_index][word_index]
                    gold_tag = mapping[int(gold[sent_index][word_index])]
                    predicted_tag = mapping[int(predicted[sent_index][word_index])]
                    if gold_tag == self.config.pad_tag:
                        continue
                    if mask[sent_index][word_index] == 0:
                        gold_tag = "<MASK>"
                        predicted_tag = "<MASK>"
                    f.write("{0}\t{1}\t{2}\n".format(token, gold_tag, predicted_tag))
                f.write("\n")

    def load_model_to_device(self):
        self.model.to(self.device)
        # TODO: Add support for parallel execution on multiple GPUs using nn.DataParallel

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def save_checkpoint(self, epoch, save_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.monitor_best
        }
        prev_checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint-epoch-{}.pth".format(epoch - 1))
        if os.path.exists(prev_checkpoint_path):
            os.remove(prev_checkpoint_path)
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint-epoch-{0}.pth".format(epoch))
        torch.save(state, checkpoint_path)
        print("Saved checkpoint: {0}".format(checkpoint_path))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, "model-best.pth")
            torch.save(state, best_path)
            print("Saved current best: model-best.pth")

    def resume_checkpoint(self):
        resume_path = self.get_model_resume_path()
        if resume_path:
            checkpoint = torch.load(resume_path, map_location=torch.device(self.device))
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.monitor_best = checkpoint["monitor_best"]
            print("Loaded checkpoint: {0}".format(resume_path))
        self.load_model_to_device()

    def get_model_resume_path(self):
        if self.config.eval == "best":
            return os.path.join(self.checkpoint_dir, "model-best.pth")
        if self.config.eval.isdigit():
            return os.path.join(self.checkpoint_dir, "checkpoint-epoch-{0}.pth".format(self.config.eval))
        return BaseExecutor.get_latest_checkpoint_path(self.checkpoint_dir)

    @staticmethod
    def get_latest_checkpoint_path(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
        max_epoch = -1
        latest_checkpoint_path = None
        for filename in checkpoint_files:
            matcher = BaseExecutor.CHECKPOINT_PATTERN.match(filename)
            if not matcher:
                continue
            curr_epoch = int(matcher.group(1))
            if max_epoch < curr_epoch:
                latest_checkpoint_path = os.path.join(checkpoint_dir, filename)
                max_epoch = curr_epoch
        return latest_checkpoint_path
