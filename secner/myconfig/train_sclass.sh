#! /bin/bash

data_root='/mnt/sdc/workspace/sec-ner/data'
out_root='/mnt/sdc/workspace/sec-ner/multirun'

datasets=("bio" "wnut")
eval_steps=(50 55)
base_models=("allenai/scibert_scivocab_uncased" "bert-base-uncased")

train_epochs=300
seq_length=256

for d in {0..1}
do
  data=${datasets[d]}
  base_model=${base_models[d]}
  eval_step=${eval_steps[d]}

  for seed in 142 242 342 442
  do
	model_type="class"
        output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
	echo "** Training SpanDetect(QA_with_charpattern) base_model=$base_model  output_dir=$output_dir"
        python main_span.py --seed $seed --model_name $model_type  --dataset_dir $data \
		  --num_train_epochs $train_epochs --max_seq_len $seq_length  \
		  --data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
		  --loss_type "dice" --query_type "question" \
		  --evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
                  --metric_for_best_model "micro_f1" --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate "1e-5" \
		  --do_train  --load_best_model_at_end  --wandb_mode "dryrun"  

	# if the best checkpoint is found, remove other checkpoints to save the disk space
	rm -r "$output_dir/runs"
	if [ -d "$output_dir/best_checkpoint" ]; then
            for chk in "$output_dir"/checkpoint-*
            do
		echo "Removing checkpoint:"$chk
                rm -r $chk
            done
	fi
   done
done
