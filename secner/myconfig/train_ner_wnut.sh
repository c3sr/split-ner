#! /bin/bash

data_root='/shared/data/jatin2/sec-ner/data'
out_root='/shared/data/jatin2/sec-ner/multirun'

train_epochs=300
seq_length=256
data="wnut"
base_model="bert-base-uncased"
eval_step_seqtag=15
eval_step_qa=80

for seed in 42
do
	model_type="ner-seq"
		output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	logfile="$output_dir/log/train/$data-$model_type-train-$seed.out"
	echo "** Training Single(SeqTagging) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"

		python main.py --seed $seed  --model_name $model_type --dataset_dir $data --num_labels 13  --tagging "bioe" \
			--num_train_epochs $train_epochs --data_root $data_root --out_root $out_root --base_model $base_model \
			--output_dir $output_dir  --max_seq_len $seq_length \
			--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
					--evaluation_strategy "steps" --eval_steps $eval_step_seqtag --save_steps $eval_step_seqtag  --save_total_limit 2 \
					--metric_for_best_model "micro_f1" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate "1e-5" \
			--run_dir $seed  --do_train  --load_best_model_at_end --wandb_mode "dryrun" 

	# if the best checkpoint is found, remove other checkpoints to save the disk space
	# rm -r "$output_dir/runs"
	# if [ -d "$output_dir/best_checkpoint" ]; then
	# 		for chk in "$output_dir"/checkpoint-*
	# 		do
	# 	echo "Removing checkpoint: $chk"
	# 			rm -r "$chk"
	# 		done
	# fi
done

for seed in 442
do
	model_type="ner-qa"
		output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	logfile="$output_dir/log/train/$data-$model_type-train-$seed.out"
	echo "** Training Single(QA)         base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
		python main_qa.py --seed $seed --model_name $model_type  --dataset_dir $data --num_labels 4 \
			--num_train_epochs $train_epochs  --data_root $data_root --out_root $out_root --base_model $base_model \
			--output_dir $output_dir --max_seq_len $seq_length  \
			--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
					--evaluation_strategy "steps" --eval_steps $eval_step_qa --save_steps $eval_step_qa  --save_total_limit 2 \
					--metric_for_best_model "micro_f1" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate "1e-5" \
			--run_dir $seed --do_train  --load_best_model_at_end  --wandb_mode "dryrun" 

	# if the best checkpoint is found, remove other checkpoints to save the disk space
	# rm -r "$output_dir/runs"
	# if [ -d "$output_dir/best_checkpoint" ]; then
	# 		for chk in "$output_dir"/checkpoint-*
	# 		do
	# 	echo "Removing checkpoint: $chk"
	# 			rm -r "$chk"
	# 		done
	# fi
done


