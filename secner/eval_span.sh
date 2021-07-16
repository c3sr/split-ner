python main_qa.py --config $1 --output "$2"
python analysis.py --datapath "../models" --dataset conll3 --model "$2" --file test --only_f1 --span_based
