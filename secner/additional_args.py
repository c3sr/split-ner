import json
import os
from enum import Enum
from typing import Optional

import dataclasses
from dataclasses import dataclass, field


@dataclass
class AdditionalArguments:
    model_name: str = field(default="ner-bert", metadata={"help": "model identifier"})
    resume: Optional[str] = field(default=None, metadata={"help": "checkpoint to resume. Starts from scratch, if None"})
    dataset_dir: str = field(default="bio", metadata={"help": "dataset dir relative to data root dir"})
    num_labels: int = field(default=4, metadata={"help": "# output labels in QA setup: 2(BO), 3(BIO), 4(BIOE)"})
    tagging: str = field(default="bio", metadata={"help": "tagging scheme (bo|bio). Not used in QA currently"})

    data_root: str = field(default="../data", metadata={"help": "data root directory"})
    out_root: str = field(default="../out", metadata={"help": "outputs root directory"})
    infer_path: str = field(default=None, metadata=
    {"help": "set to predictions file of span detector in predict mode, relative to predictions dir"})
    train_path: str = field(default="train.tsv", metadata={"help": "train file path relative to data root"})
    dev_path: str = field(default="dev.tsv", metadata={"help": "dev file path relative to data root"})
    test_path: str = field(default="test.tsv", metadata={"help": "test file path relative to data root"})
    tag_vocab_path: str = field(default="tag_vocab.txt", metadata={"help": "tag vocab file path relative to data root"})
    pos_tag_vocab_path: str = field(default="pos_tag_vocab.txt", metadata={"help": "pos tag vocab file path"})
    dep_tag_vocab_path: str = field(default="dep_tag_vocab.txt", metadata={"help": "dep tag vocab file path"})
    tag_names_path: str = field(default="tag_names.txt", metadata={"help": "tag to its english name mapping"})

    pad_tag: str = field(default="[PAD]", metadata={"help": "symbol for padding token/tag"})
    none_tag: str = field(default="O", metadata={"help": "symbol for NONE tag"})

    max_seq_len: int = field(default=128, metadata={"help": "maximum sequence length"})
    punctuation_handling: str = field(default="none", metadata=
    {"help": "add one-hot entry to represent if token is punct (none|type1|type2|type1-and)"})
    word_type_handling: str = field(default="none", metadata={"help": "add word type (token lower/digit)(none|1hot)"})
    base_model: str = field(default="bert-base-uncased", metadata={"help": "base pretrained model for training"})
    freeze_bert: bool = field(default=False, metadata={"help": "freeze base bert model's parameters during training"})
    model_mode: str = field(default="std", metadata={"help": "model mode (std|crf|bidaf|char|roberta_std)"})
    loss_type: str = field(default="ce", metadata={"help": "loss type in 'std' mode (ce|ce_wt|ce_punct|dice)"})
    use_char_cnn: str = field(default="none", metadata=
    {"help": "use char CNN (none|char|pattern|flair|both|both-flair)"})
    token_type: str = field(default="text", metadata={"help": "token type used with CNN/pattern etc. (text|sub_text)"})
    pattern_type: str = field(default="0", metadata={"help": "pattern type (0|1)"})
    use_end_cnn: bool = field(default=False, metadata={"help": "use end CNN"})
    add_qa_helper_sentence: bool = field(default=False, metadata={"help": "add helper sentence with known entities"})
    use_pattern: str = field(default="none", metadata={"help": "use patterns over actual mentions (none|only|both)"})
    query_type: str = field(default="question", metadata=
    {"help": "query type for entities fed in QA model (question|question2)"})
    detect_spans: bool = field(default=False, metadata={"help": "in QA mode with this set, we detect only spans"})
    char_emb_dim: int = field(default=16, metadata={"help": "char embedding dimension (input channels to char CNN)"})
    lstm_hidden_dim: int = field(default=256, metadata={"help": "pattern LSTM hidden dim"})
    lstm_num_layers: int = field(default=1, metadata={"help": "pattern LSTM: no. of layers"})
    cnn_num_filters: int = field(default=16, metadata={"help": "# char CNN filters"})
    cnn_kernel_size: int = field(default=5, metadata={"help": "char CNN kernel size"})
    end_cnn_channels: int = field(default=8, metadata={"help": "end CNN output channels"})
    cnn_dropout_rate: float = field(default=0.3, metadata={"help": "char CNN dropout rate"})
    second_classifier_hidden_sz: int = field(default=0, metadata=
    {"help": "use 2nd classifier layer with supplied hidden size (use 0 for no second layer)"})
    use_head_mask: bool = field(default=False, metadata={"help": "use only head sub-token's output from BERT"})
    data_pos_dep: bool = field(default=True, metadata={"help": "dataset has labeled POS/DEP tags"})
    use_pos_tag: bool = field(default=False, metadata={"help": "use 1-hot POS tag vectors for tokens"})
    use_dep_tag: bool = field(default=False, metadata={"help": "use 1-hot DEP-parse tag vectors for tokens"})
    use_main_lstm: bool = field(default=False, metadata={"help": "use bi-LSTM as the main LM"})
    filter_tags: object = field(default=None, metadata={"help": "consider only given output tags"})
    split_tags: bool = field(default=False, metadata={"help": "split big categories into smaller ones"})
    gold_span_inp: str = field(default="none", metadata={"help": "provide gold span as input (none|simple|label)"})
    wandb_mode: str = field(default="run", metadata={"help": "can enable/disable wandb online sync (run/dryrun)"})
    debug_mode: bool = field(default=False, metadata={"help": "truncate dataset for faster debugging"})

    def __post_init__(self):
        self.run_root = os.path.join(self.out_root, self.dataset_dir, self.model_name)
        if self.resume:
            self.resume = os.path.join(self.run_root, "checkpoints", "checkpoint-{0}".format(self.resume))
        self.wandb_dir = self.run_root
        self.predictions_dir = os.path.join(self.run_root, "predictions")

        if self.infer_path:
            self.infer_path = os.path.join(self.predictions_dir, self.infer_path)

        self.abs_dataset_dir = os.path.join(self.data_root, self.dataset_dir)
        self.train_path = os.path.join(self.abs_dataset_dir, self.train_path)
        self.dev_path = os.path.join(self.abs_dataset_dir, self.dev_path)
        self.test_path = os.path.join(self.abs_dataset_dir, self.test_path)
        self.tag_vocab_path = os.path.join(self.abs_dataset_dir, self.tag_vocab_path)
        self.pos_tag_vocab_path = os.path.join(self.abs_dataset_dir, self.pos_tag_vocab_path)
        self.dep_tag_vocab_path = os.path.join(self.abs_dataset_dir, self.dep_tag_vocab_path)
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
