#! /bin/bash

data_root='/mnt/sdc/workspace/sec-ner/data'
out_root='/mnt/sdc/workspace/sec-ner/multirun'

datasets=("wnut" "bio")
num_labels=(13 33 17)
eval_steps=(55 50 600)
base_models=("bert-base-uncased" "allenai/scibert_scivocab_uncased" "bert-base-uncased")

train_epochs=300
seq_length=256

for d in {0..1}
do
  data=${datasets[d]}
  base_model=${base_models[d]}
  num_label=${num_labels[d]}
  eval_step=${eval_steps[d]}

  for seed in 142  242 342 442  
  do
	model_type="ner-seq"
        output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	logfile="./log/train/$data-$model_type-train-$seed.out"
	echo "** Training Single(SeqTagging) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"

        python main.py --seed $seed  --model_name $model_type --dataset_dir $data --num_labels $num_label  --tagging "bioe" \
		  --num_train_epochs $train_epochs --data_root $data_root --out_root $out_root --base_model $base_model \
		  --output_dir $output_dir  --max_seq_len $seq_length \
		  --token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
                  --evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
                  --metric_for_best_model "micro_f1" --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate "1e-5" \
		  --do_train  --load_best_model_at_end --wandb_mode "dryrun" 

	# if the best checkpoint is found, remove other checkpoints to save the disk space
	rm -r "$output_dir/runs"
	if [ -d "$output_dir/best_checkpoint" ]; then
            for chk in "$output_dir"/checkpoint-*
            do
		echo "Removing checkpoint: $chk"
                rm -r "$chk"
            done
	fi


	model_type="ner-qa"
        output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	logfile="./log/train/$data-$model_type-train-$seed.out"
	echo "** Training Single(QA)         base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
        python main_qa.py --seed $seed --model_name $model_type  --dataset_dir $data --num_labels $num_label \
		  --num_train_epochs $train_epochs  --data_root $data_root --out_root $out_root --base_model $base_model \
		  --output_dir $output_dir --max_seq_len $seq_length  \
		  --token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
                  --evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
                  --metric_for_best_model "micro_f1" --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate "1e-5" \
		  --do_train  --load_best_model_at_end  --wandb_mode "dryrun" 

	# if the best checkpoint is found, remove other checkpoints to save the disk space
	rm -r "$output_dir/runs"
	if [ -d "$output_dir/best_checkpoint" ]; then
            for chk in "$output_dir"/checkpoint-*
            do
		echo "Removing checkpoint: $chk"
                rm -r "$chk"
            done
	fi
    done
done
