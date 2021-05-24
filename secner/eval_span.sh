python main_qa.py --config $1 --output "$2"
python analysis.py --data bio --model "$2" --file test --only_f1 --span_based
