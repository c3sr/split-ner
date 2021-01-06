# NER on Security Data

Simple NER model framework designed with the target of low-resource extractions on security data.
## Usage
Installation:
```commandline
pip install -r requirements.txt
pip install -e .
```
Run training, in corresponding config.json, do:
For evaluation, in config.json, do:
 ```json
{
 "do_train": true,
 "eval": null
}
```
Then,
```commandline
cd secner
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config.json
```
For evaluation on checkpoint (say, ```4840```), in config.json, do:
```json
{
 "do_train": false,
 "eval": "4840"
}
```