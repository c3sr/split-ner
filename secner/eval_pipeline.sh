python main_span.py --config $1
python analysis.py --modelpath "../models" --dataset $2 --model $3  --file infer --only_f1
