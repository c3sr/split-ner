python main_qa.py $1
python analysis.py --experiment_dir "../emnlp" --dataset $2 --model "$3" --file test --span_based
