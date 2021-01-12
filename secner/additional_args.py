import json
import os
from enum import Enum
from typing import Optional

import dataclasses
from dataclasses import dataclass, field


@dataclass
class AdditionalArguments:
    model_name: str = field(default="ner-bert", metadata={"help": "model identifier"})
    resume: Optional[str] = field(default=None,
                                  metadata={"help": "checkpoint to resume training. Starts from scratch, if None"})
    dataset_dir: str = field(default="bio", metadata={"help": "dataset dir relative to data root dir"})
    num_labels: int = field(default=33, metadata={"help": "# output labels in NER dataset"})

    data_root: str = field(default="../data", metadata={"help": "data root directory"})
    out_root: str = field(default="../out", metadata={"help": "outputs root directory"})
    train_path: str = field(default="train.tsv", metadata={"help": "train file path relative to data root"})
    dev_path: str = field(default="dev.tsv", metadata={"help": "dev file path relative to data root"})
    test_path: str = field(default="test.tsv", metadata={"help": "test file path relative to data root"})
    tag_vocab_path: str = field(default="tag_vocab.txt",
                                metadata={"help": "tag vocab file path relative to data root"})
    tag_names_path: str = field(default="tag_names.txt",
                                metadata={"help": "tag to its english name (text) mapping"})

    pad_tag: str = field(default="[PAD]", metadata={"help": "symbol for padding token/tag"})
    none_tag: str = field(default="O", metadata={"help": "symbol for NONE tag"})

    max_seq_len: int = field(default=128, metadata={"help": "maximum sequence length"})
    base_model: str = field(default="bert-base-uncased", metadata={"help": "base pretrained model for training"})
    freeze_bert: bool = field(default=False, metadata={"help": "freeze base bert model's parameters during training"})
    query_type: str = field(default="question", metadata={"help": "query type for entities fed in QA model"})

    def __post_init__(self):
        self.run_root = os.path.join(self.out_root, self.dataset_dir, self.model_name)
        if self.resume:
            self.resume = os.path.join(self.run_root, "checkpoints", "checkpoint-{0}".format(self.resume))
        self.wandb_dir = self.run_root
        self.predictions_dir = os.path.join(self.run_root, "predictions")

        self.abs_dataset_dir = os.path.join(self.data_root, self.dataset_dir)
        self.train_path = os.path.join(self.abs_dataset_dir, self.train_path)
        self.dev_path = os.path.join(self.abs_dataset_dir, self.dev_path)
        self.test_path = os.path.join(self.abs_dataset_dir, self.test_path)
        self.tag_vocab_path = os.path.join(self.abs_dataset_dir, self.tag_vocab_path)
        self.tag_names_path = os.path.join(self.abs_dataset_dir, self.tag_names_path)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)
