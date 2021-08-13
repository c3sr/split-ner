python main_span.py --config $1
python analysis.py --datapath "../models" --dataset $3 --model $2 --file test --only_f1
