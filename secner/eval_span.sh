python main_qa.py --config $1
python analysis.py --modelpath "../emnlp" --dataset $2 --model "$3" --file test --only_f1 --span_based
