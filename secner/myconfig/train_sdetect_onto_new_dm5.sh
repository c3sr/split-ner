#! /bin/bash

data_root='/shared/data/jatin2/sec-ner/data'
out_root='/shared/data/jatin2/sec-ner/multirun'

train_epochs=300
seq_length=512
data="onto_final"
base_model="roberta-base"
eval_step=500
batch_size=16

# STOPPED PARTIALLY (current best: 89.93 - overall)
# for seed in 242
# 	do
# 	model_type="span-seq-new"
# 	output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
# 	logfile="./log/train/$data-$model_type-train-$seed.out"
# 	echo "** Training SpanDetect(SeqTag) base_model=$base_model data_set = $data  output_dir=$output_dir  log_file=$logfile" 
# 	python main.py --seed $seed --model_name $model_type --tagging "bioe"  --dataset_dir $data --num_labels 37 \
# 		--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
# 		--num_train_epochs $train_epochs --max_seq_len $seq_length --model_mode "roberta_std"  \
# 		--data_root $data_root --out_root $out_root --base_model $base_model  --output_dir $output_dir \
# 		--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
# 				--metric_for_best_model "micro_f1" --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --learning_rate "1e-5" \
# 		--run_dir $seed --do_eval  --load_best_model_at_end --wandb_mode "dryrun" --resume "64000" \
# 		--detect_spans
# 	done

for seed in 42
	do
	model_type="class-ce-new"
	output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	echo "** Training SpanClass(CE) base_model=$base_model  output_dir=$output_dir"
	python main_span.py --seed $seed --model_name $model_type  --dataset_dir $data \
		--num_train_epochs $train_epochs --max_seq_len $seq_length --model_mode "roberta_std" \
		--data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
		--loss_type "ce" --query_type "question" \
		--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
		--metric_for_best_model "micro_f1" --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --learning_rate "1e-5" \
		--run_dir $seed --do_train  --load_best_model_at_end  --wandb_mode "dryrun"  
	done
