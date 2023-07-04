#! /bin/bash

data_root='/shared/data/jatin2/sec-ner/data'
out_root='/shared/data/jatin2/sec-ner/multirun'

train_epochs=300
seq_length=256
data="wnut"
base_model="bert-base-uncased"
eval_step=15

for seed in 42
	do
	model_type="span-seq"
		output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	logfile="./log/train/$data-$model_type-train-$seed.out"
	echo "** Training SpanDetect(SeqTag) base_model=$base_model data_set = $data  output_dir=$output_dir  log_file=$logfile" 
		python main.py --seed $seed --model_name $model_type --tagging "bioe"  --dataset_dir $data --num_labels 13 \
			--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
			--num_train_epochs $train_epochs --max_seq_len $seq_length \
			--data_root $data_root --out_root $out_root --base_model $base_model  --output_dir $output_dir \
			--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
					--metric_for_best_model "micro_f1" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate "1e-5" \
			--my_seed $seed --do_train  --load_best_model_at_end --wandb_mode "dryrun" \
			--detect_spans
	done

	# if the best checkpoint is found, remove other checkpoints to save the disk space
	# rm -r "$output_dir/runs"
	# if [ -d "$output_dir/best_checkpoint" ]; then
	# 		for chk in "$output_dir"/checkpoint-*
	# 		do
	# 	echo "Removing checkpoint: $chk"
	# 			rm -r "$chk"
	# 		done
	# fi

for seed in 142
	do
	model_type="span-qa"
		output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	logfile="./log/train/$data-$model_type-train-$seed.out"
	echo "** Training SpanDetect(QA_with_charpattern) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
		python main_qa.py --seed $seed --model_name $model_type  --dataset_dir $data --num_labels 4 \
			--num_train_epochs $train_epochs --max_seq_len $seq_length  \
			--data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
			--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
			--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
					--metric_for_best_model "micro_f1" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate "1e-5" \
			--my_seed $seed --do_train  --load_best_model_at_end  --wandb_mode "dryrun"  \
			--detect_spans
	done
	# if the best checkpoint is found, remove other checkpoints to save the disk space
	# rm -r "$output_dir/runs"
	# if [ -d "$output_dir/best_checkpoint" ]; then
	# 		for chk in "$output_dir"/checkpoint-*
	# 		do
	# 	echo "Removing checkpoint: $chk"
	# 			rm -r "$chk"
	# 		done
	# fi

for seed in 142
	do
	model_type="span-qa-nocharpattern"
		output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	logfile="./log/train/$data-$model_type-train-$seed.out"
	echo "** Training SpanDetect(QA_no_charpattern) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
		python main_qa.py --seed $seed --model_name $model_type  --dataset_dir $data --num_labels 4 \
			--num_train_epochs $train_epochs --max_seq_len $seq_length  \
			--data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir \
			--use_char_cnn "none" --query_type "question4" \
			--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
					--metric_for_best_model "micro_f1" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate "1e-5" \
			--my_seed $seed --do_train --load_best_model_at_end  --wandb_mode "dryrun" \
			--detect_spans

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

for seed in 42
do
        model_type="class"
        output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
        echo "** Training SpanClass(CE) base_model=$base_model  output_dir=$output_dir"
        python main_span.py --seed $seed --model_name $model_type  --dataset_dir $data \
                        --num_train_epochs $train_epochs --max_seq_len $seq_length \
                        --data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
                        --loss_type "ce" --query_type "question" \
                        --evaluation_strategy "steps" --eval_steps 10 --save_steps 10  --save_total_limit 2 \
                        --metric_for_best_model "micro_f1" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate "1e-5" \
                        --my_seed $seed --do_train  --load_best_model_at_end  --wandb_mode "dryrun"  

        # if the best checkpoint is found, remove other checkpoints to save the disk space
        # rm -r "$output_dir/runs"
        # if [ -d "$output_dir/best_checkpoint" ]; then
        #         for chk in "$output_dir"/checkpoint-*
        #         do
        #         echo "Removing checkpoint:"$chk
        #         rm -r $chk
        #         done
        # fi
done