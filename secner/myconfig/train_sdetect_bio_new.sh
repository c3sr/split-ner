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
# 	model_type="span-qa-pattern"
# 	output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
# 	logfile="./log/train/$data-$model_type-train-$seed.out"
# 	echo "** Training SpanDetect(QA_with_pattern) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
# 	python main_qa.py --seed $seed --model_name $model_type  --dataset_dir $data --num_labels 4 \
# 		--num_train_epochs $train_epochs --max_seq_len $seq_length  \
# 		--data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
# 		--token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "pattern" --query_type "question4" \
# 		--evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
# 				--metric_for_best_model "micro_f1" --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --learning_rate "1e-5" \
# 		--my_seed $seed --do_train  --load_best_model_at_end  --wandb_mode "dryrun"  \
# 		--detect_spans
# 	done

