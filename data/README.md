# Data Directory

All datasets used for experiments are supposed to be present in this directory by default. Each dataset must have the following files (see the `dummy` dataset provided as an example):
* `train.tsv`: Train dataset (in below mentioned format)
* `dev.tsv`: Dev dataset (in below mentioned format)
* `test.tsv`: Test dataset (in below mentioned format)
* `tag_vocab.txt`: Vocabulary of NER tags (BIO tagging scheme by default)
* `tag_names.txt`: English names for each NER tag. For example `PER` means `person`. This helps in generating questions for QA-based models
* `pos_tag_vocab.txt`: possible Part-of-Speech (POS) tags for words among sentences in the dataset (used for some experiments)
* `dep_tag_vocab.txt`: possible Dependency Parse tags for words among sentences in the dataset (used for some experiments)

The POS tags and Dependency Parse tags (Dep Tag) are not ultimately used by the model proposed in the final version of the paper but the code provides facility to feed these as additional input features to the model and hence, for parsing, these files are required. 

### Dataset Format

* Every token for every sentence in the dataset comes in a new line
* Every line has tab-separated format: `<Token>\t<POS tag>\t<Dep Tag>\t<NER Tag>\n`
* Between every two sentences in the dataset is a blank new line
