python main.py $1 
python analysis.py --experiment_dir "../models" --dataset $2 --model "$3" --file test --span_based
