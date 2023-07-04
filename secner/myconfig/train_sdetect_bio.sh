#! /bin/bash

data_root='/shared/data/jatin2/sec-ner/data'
out_root='/shared/data/jatin2/sec-ner/multirun'

train_epochs=300
seq_length=256
data="bio"
base_model="allenai/scibert_scivocab_uncased"
eval_step=16
batch_size=32

# for seed in 42
# 	do
# 	model_type="ner-seq"
# 	output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
# 	logfile="$output_dir/log/train/$data-$model_type-train-$seed.out"
# 	echo "** Training Single(SeqTagging) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
# 	python main.py --seed $seed  --model_name $model_type --dataset_dir $data --num_labels 33  --tagging "bioe" \
# 		--num_train_epochs $train_epochs --data_root $data_root --out_root $out_root --base_model $base_model \
# 		--output_dir $output_dir  --max_seq_len $seq_length \
# 		--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
# 				--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
# 				--metric_for_best_model "micro_f1" --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --learning_rate "1e-5" \
# 		--my_seed $seed  --do_train  --load_best_model_at_end --wandb_mode "dryrun" 
# 	done

# for seed in 42
# 	do
# 	model_type="span-seq"
# 	output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
# 	logfile="./log/train/$data-$model_type-train-$seed.out"
# 	echo "** Training SpanDetect(SeqTag) base_model=$base_model data_set = $data  output_dir=$output_dir  log_file=$logfile" 
# 	python main.py --seed $seed --model_name $model_type --tagging "bioe"  --dataset_dir $data --num_labels 33 \
# 		--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
# 		--num_train_epochs $train_epochs --max_seq_len $seq_length \
# 		--data_root $data_root --out_root $out_root --base_model $base_model  --output_dir $output_dir \
# 		--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
# 				--metric_for_best_model "micro_f1" --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --learning_rate "1e-5" \
# 		--my_seed $seed --do_train  --load_best_model_at_end --wandb_mode "dryrun" \
# 		--detect_spans
# 	done

# for seed in 42
# 	do
# 	model_type="span-qa"
# 	output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
# 	logfile="./log/train/$data-$model_type-train-$seed.out"
# 	echo "** Training SpanDetect(QA_with_charpattern) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
# 	python main_qa.py --seed $seed --model_name $model_type  --dataset_dir $data --num_labels 4 \
# 		--num_train_epochs $train_epochs --max_seq_len $seq_length  \
# 		--data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
# 		--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
# 		--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
# 				--metric_for_best_model "micro_f1" --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --learning_rate "1e-5" \
# 		--my_seed $seed --do_train  --load_best_model_at_end  --wandb_mode "dryrun"  \
# 		--detect_spans --resume "1552"
# 	done

# for seed in 42
# 	do
# 	model_type="span-qa-nocharpattern-querytype3"
# 		output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
# 	logfile="./log/train/$data-$model_type-train-$seed.out"
# 	echo "** Training SpanDetect(QA_no_charpattern_querytype3) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
# 	python main_qa.py --seed $seed --model_name $model_type  --dataset_dir $data --num_labels 4 \
# 		--num_train_epochs $train_epochs --max_seq_len $seq_length  \
# 		--data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir \
# 		--use_char_cnn "none" --query_type "question3" \
# 		--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
# 				--metric_for_best_model "micro_f1" --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --learning_rate "1e-5" \
# 		--my_seed $seed --do_train --load_best_model_at_end  --wandb_mode "dryrun" --resume "8040" \
# 		--detect_spans
# 	done

# for seed in 42
# 	do
# 	model_type="span-qa-char"
# 	output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
# 	logfile="./log/train/$data-$model_type-train-$seed.out"
# 	echo "** Training SpanDetect(QA_with_char) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
# 	python main_qa.py --seed $seed --model_name $model_type  --dataset_dir $data --num_labels 4 \
# 		--num_train_epochs $train_epochs --max_seq_len $seq_length  \
# 		--data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
# 		--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "char" --query_type "question4" \
# 		--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
# 				--metric_for_best_model "micro_f1" --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --learning_rate "1e-5" \
# 		--my_seed $seed --do_train  --load_best_model_at_end  --wandb_mode "dryrun"  \
# 		--detect_spans
# 	done

# NOT DONE YET.. TAKES TOO MUCH MEMORY
for seed in 42
	do
	model_type="span-qa-char-pattern-pos"
	output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	logfile="./log/train/$data-$model_type-train-$seed.out"
	echo "** Training SpanDetect(QA_with_charpattern_pos) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
	python main_qa.py --seed $seed --model_name $model_type  --dataset_dir $data --num_labels 4 \
		--num_train_epochs $train_epochs --max_seq_len $seq_length  \
		--data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
		--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
		--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
				--metric_for_best_model "micro_f1" --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --learning_rate "1e-5" \
		--my_seed $seed --do_train  --load_best_model_at_end  --wandb_mode "dryrun"  \
		--detect_spans --use_pos_tag
	done