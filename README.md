# SplitNER

Named Entity Recognition via Two Question-Answering-based Classifications

## Installation
```shell script
pip install -r requirements.txt
pip install -e .
```

## Execution

All model training / evaluations happens via run arguments specified in config (JSON) files. Since only code is shared, we provide a small ```dummy``` dataset here for ease of explanation. Any new dataset should be formatted and used in the same way. 

The dummy data is actually a small randomly sampled set of sentences from ```WNUT17``` corpus [website](https://noisy-text.github.io/2017/emerging-rare-entities.html) and is just being used for illustration. This should help the reader get the code up and runing, understand input/output data format and configure different train/eval scripts.

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
 "resume": null,
}
```

For resuming training from the latest checkpoint, find the last saved checkpoint from the checkpoint dir (default path: ```out/dummy/<model name>/<run name>/checkpoints```). Say last checkpoint was ```checkpoint-4840```, then update in config:
```json
"resume": "4840",
```

### Evaluation

* Reports the F1 metrics (token-level) for specified model checkpoint on test set
* Saves model predictions for train/dev/test sets under ```out/dummy/<model name>/<run name>/predictions```

Whenever evaluating a model on saved checkpoint (say, ```4840```), make sure to set in its config file:
```json
{
 "do_train": false,
 "resume": "4840",
}
```

### Mention-Level Metrics

The following script takes the output predictions from the model and calculates mention-level metrics (most importantly Micro-F1 Score) as is most commonly reported in literature. If output predictions are in file ```<repo root>/out/dummy/<model name>/<run name>/predictions/test.tsv``` then,

```shell script
python analysis.py --experiment_dir out --dataset dummy --model <model name> --run_dir <run name> --file test
```

For getting metrics for span detection only, add flag ```--span_based```. Example, for ```spandetect``` model:

```shell script
python analysis.py --experiment_dir out --dataset dummy --model spandetect --run_dir <run name> --file test --span_based
```

## Run Commands

Move to the working directory
```shell script
cd splitner
```

#### Span Detection Module Training / Evaluation

Make sure config file has ```"detect_spans": true```.  Example run command:

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_qa.py ../config/dummy/spandetect.json
```

#### Span Classification Module Training / Evaluation

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_span.py ../config/dummy/spanclass-dice.json
```

#### SplitNER Pipeline
Once the above models are individually trained and evaluated, copy ```test.tsv``` from ```predictions``` folder of Span Detector into ```predictions``` folder for Span Classification model. Rename this file as say, ```infer_inp.tsv```. Specify this file name in the config for span classifier (as shown below). Specify an output file path say, ```infer.tsv```.

```json
"infer_inp_path": "infer_inp.tsv",
"infer_out_path": "infer.tsv",
```

Run evaluation once again.

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_span.py ../config/dummy/spanclass-dice.json
```

The final outputs whill be in file ```infer.tsv``` under the ```predictions``` folder of Span Classification model. To get the mention-level F1-score metrics over your outputs, run:

```shell script
python analysis.py --experiment_dir out --dataset dummy --model spanclass-dice --run_dir <run name> --file infer
```

#### Baseline: Single-QA Training / Evaluation

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main_qa.py ../config/dummy/single-qa.json
```

#### Baseline: Single-SeqTagging Training / Evaluation

```shell script
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
