import abc
import os
import re

import numpy as np
import torch

from secner.utils import get_compute_device, set_absolute_paths


class BaseExecutor:
    CHECKPOINT_PATTERN = re.compile("checkpoint-epoch-(\\d+)\\.pth")

    def __init__(self, config):
        self.config = config
        self.device = get_compute_device(self.config.use_cpu)
        print("using device: {0}".format(self.device))

        set_absolute_paths(self.config)
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        if not os.path.exists(self.config.data.out_dir):
            os.makedirs(self.config.data.out_dir)

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        self.train_data_loader = None
        self.dev_data_loader = None
        self.test_data_loader = None

        self.model = None
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
        # print(self.model)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        num_params = sum([np.prod(p.size()) for p in trainable_params])
        print("No. of trainable parameters: {0}".format(num_params))

    def run(self):
        self.resume_checkpoint()
        self.print_model_summary()

        if self.config.eval != "none":
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

    @abc.abstractmethod
    def train_epoch(self, epoch):
        pass

    def evaluate(self):
        train_outfile = os.path.join(self.config.data.out_dir, "train.tsv")
        dev_outfile = os.path.join(self.config.data.out_dir, "dev.tsv")
        test_outfile = os.path.join(self.config.data.out_dir, "test.tsv")
        last_trained_epoch = self.start_epoch - 1
        train_evaluator = self.evaluate_epoch(self.train_data_loader, epoch=last_trained_epoch, prefix="TRAIN",
                                              outfile=train_outfile)
        dev_evaluator = self.evaluate_epoch(self.dev_data_loader, epoch=last_trained_epoch, prefix="DEV",
                                            outfile=dev_outfile)
        test_evaluator = self.evaluate_epoch(self.test_data_loader, epoch=last_trained_epoch, prefix="TEST",
                                             outfile=test_outfile)

        return train_evaluator, dev_evaluator, test_evaluator

    @abc.abstractmethod
    def evaluate_epoch(self, data_loader, epoch, prefix, outfile=None):
        pass

    def print_outputs(self, corpus, gold, predicted, offsets, mapping, outfile):
        data = []
        none_tag = self.config.none_tag
        for i in range(len(corpus)):
            data.append([[text, none_tag, none_tag] for text in corpus[i] if text != self.config.pad_tag])
            ptr = 0
            for j in range(len(offsets[i])):
                if offsets[i][j] != ptr:
                    continue
                data[i][ptr][1] = mapping[gold[i][j]]
                data[i][ptr][2] = mapping[predicted[i][j]]
                ptr += 1

        print("Outputs published in file: {0}".format(outfile))
        with open(outfile, "w") as f:
            # f.write("Token\tGold\tPredicted\n")
            for sent in data:
                for word in sent:
                    f.write("{0}\t{1}\t{2}\n".format(word[0], word[1], word[2]))
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
        prev_checkpoint_path = os.path.join(self.config.checkpoint_dir, "checkpoint-epoch-{}.pth".format(epoch - 1))
        if os.path.exists(prev_checkpoint_path):
            os.remove(prev_checkpoint_path)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "checkpoint-epoch-{0}.pth".format(epoch))
        torch.save(state, checkpoint_path)
        print("Saved checkpoint: {0}".format(checkpoint_path))
        if save_best:
            best_path = os.path.join(self.config.checkpoint_dir, "model-best.pth")
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
            return os.path.join(self.config.checkpoint_dir, "model-best.pth")
        if self.config.eval.isdigit():
            return os.path.join(self.config.checkpoint_dir, "checkpoint-epoch-{0}.pth".format(self.config.eval))
        return BaseExecutor.get_latest_checkpoint_path(self.config.checkpoint_dir)

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
