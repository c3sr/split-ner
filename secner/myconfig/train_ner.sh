#! /bin/bash

data_root='/mnt/sdc/workspace/sec-ner/data'
out_root='/mnt/sdc/workspace/sec-ner/multirun'

datasets=("wnut" "bio" "security")
num_labels=(13 33 17)
base_models=("bert-base-uncased" "allenai/scibert_scivocab_uncased" "bert-base-uncased")

train_epochs=300
seq_length=256

for d in {0..2}
do
  data=${datasets[d]}
  num_label=${num_labels[d]}
  base_model=${base_models[d]}

  for i in {1..5}
  do
	model_type="ner-seq"
        output_dir="$out_root/$data/$model_type/run-$i/checkpoints"
	logfile="./log/train/$data-$model_type-train-$i.out"
	echo "** Training Single(SeqTagging) base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
        python main.py --seed "-1" --model_name $model_type --dataset_dir $data --num_labels $num_label \
		  --num_train_epochs $train_epochs --data_root $data_root --out_root $out_root --base_model $base_model \
		  --output_dir $output_dir  --max_seq_len $seq_length \
                  --evaluation_strategy "epoch" --save_strategy "epoch" --logging_steps 100 --save_total_limit 2 \
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
        output_dir="$out_root/$data/$model_type/run-$i/checkpoints"
	logfile="./log/train/$data-$model_type-train-$i.out"
	echo "** Training Single(QA)         base_model=$base_model  output_dir=$output_dir  log_file=$logfile"
        python main_qa.py --seed "-1" --model_name $model_type  --dataset_dir $data --num_labels $num_label \
		  --num_train_epochs $train_epochs  --data_root $data_root --out_root $out_root --base_model $base_model \
		  --output_dir $output_dir --max_seq_len $seq_length  \
		  --token_type "sub_text" --char_emb_dim 50 --pattern_type "3" --use_char_cnn "both" --query_type "question4" \
                  --evaluation_strategy "epoch" --save_strategy "epoch" --logging_steps 100 --save_total_limit 2 \
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
