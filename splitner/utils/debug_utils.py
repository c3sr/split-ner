import torch
from torch.utils.data import SequentialSampler
from transformers.trainer import Trainer
from typing import Optional


# use learning_rate=0.001 to replicate old split-ner model behavior
class SequentialTrainer(Trainer):
    def __init__(self, model, args, data_collator, train_dataset, eval_dataset, tokenizer, compute_metrics):
        super(SequentialTrainer, self).__init__(model=model,
                                                args=args,
                                                data_collator=data_collator,
                                                train_dataset=train_dataset,
                                                eval_dataset=eval_dataset,
                                                tokenizer=tokenizer,
                                                compute_metrics=compute_metrics)

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        return SequentialSampler(self.train_dataset)
