# SplitNER

Named Entity Recognition via Two Question-Answering-based Classifications

## Installation
```commandline
pip install -r requirements.txt
pip install -e .
```

## Execution

All model training / evaluations happens via run arguments specified in config (JSON) files. Since only code is shared, we provide a small ```dummy``` dataset here for ease of explanation. Any new dataset should be formatted and used in the same way. 

The dummy data is actually a small randomly sampled set of sentences from ```WNUT17``` corpus [website](https://noisy-text.github.io/2017/emerging-rare-entities.html) and is just being used for illustration.

Citation:

Leon Derczynski, Eric Nichols, Marieke van Erp, Nut Limsopatham (2017) "Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition", in Proceedings of the 3rd Workshop on Noisy, User-generated Text.

### Training

* Runs training for specified number of epochs
* Keeps logging training progress, best F1 metrics so far among other things
* Keeps evaluating on dev set after specified steps and reports the F1 metrics
* At any given point of time, makes sure to save both best and latest checkpoints

Whenever training a model, make sure to set in its config file:
 ```json
{
 "do_train": true,
 "resume": null
}
```

### Evaluation

* Reports the F1 metrics for specified model checkpoint on train/dev/test sets
* Saves model predictions for train/dev/test sets under ```out/dummy/<model name>/predictions```

Whenever evaluating a model on saved checkpoint (say, ```4840```), make sure to set in its config file:
```json
{
 "do_train": false,
 "resume": "4840"
}
```

## Run Commands

Move to the working directory
```commandline
cd splitner
```

#### Span Detection Module Training / Evaluation

Make sure config file has ```"detect_spans": true```.  Example run command:

```commandline
CUDA_VISIBLE_DEVICES=0,1 python main_qa.py ../config/dummy/spandetect.json
```

#### Span Classification Module Training / Evaluation

```commandline
CUDA_VISIBLE_DEVICES=0,1 python main_span.py ../config/dummy/spanclass-dice.json
```

#### SplitNER Pipeline
Once the above models are individually trained and evaluated, copy ```test.tsv``` from ```predictions``` folder of Span Detector into ```predictions``` folder for Span Classification model. Rename this file as say, ```infer_inp.tsv```. Specify this file name in the config for span classifier (as shown below) and run its evaluation once again.

```json
"infer_path": "infer_inp.tsv"
```

The final outputs are generated always in file ```infer.tsv``` under the ```predictions``` folder of Span Classification model. To get the overall F1-score metrics run:

```shell script
python analysis.py --dataset dummy --model spanclass-dice --file infer --only_f1
```

#### Baseline: Single-QA Training / Evaluation

```commandline
CUDA_VISIBLE_DEVICES=0,1 python main_qa.py ../config/dummy/single-qa.json
```

#### Baseline: Single-SeqTagging Training / Evaluation

```commandline
CUDA_VISIBLE_DEVICES=0,1 python main.py ../config/dummy/single-seqtag.json
```

### Reproducing Results

Experiments are performed on 4 datsets: ```BioNLP13CG```, ```CyberThreats```, ```OntoNotes5.0```, ```WNUT17```. Config files are similar to the ones for ```dummy``` dataset with the below differences.

#### Max Sequence Length

* Set ```"max_seq_len": 512``` for ```OntoNotes5.0```
* Set ```"max_seq_len": 256``` for other datasets

#### Base Model

* Set ```"base_model": "bert-base-uncased"``` for ```WNUT17``` and ```CyberThreats```
* Set ```"base_model": "allenai/scibert_scivocab_uncased"``` for ```BIONLP13CG```
* Set ```"base_model": "roberta-base"``` and ```"model_mode": "roberta_std"``` for ```OntoNotes5.0```
