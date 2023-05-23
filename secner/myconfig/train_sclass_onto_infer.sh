#! /bin/bash

data_root='/shared/data/jatin2/sec-ner/data'
out_root='/shared/data/jatin2/sec-ner/multirun'

train_epochs=10
seq_length=512

data="onto_final"
base_model="roberta-base"
eval_step=2000
model_type="class"

for seed in 142 242 342 442
do
        infer="span-seq"
        infer_inp="span-seq-inp.tsv"
        infer_out="span-seq-out"
        output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
        echo "** $infer Training SpanDetect(QA_with_charpattern) base_model=$base_model  output_dir=$output_dir"
        python main_span.py --seed $seed --model_name $model_type  --dataset_dir $data \
                        --num_train_epochs $train_epochs --max_seq_len $seq_length --model_mode "roberta_std" \
                        --data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
                        --loss_type "dice" --query_type "question" \
                        --evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
                        --metric_for_best_model "micro_f1" --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --learning_rate "5e-5" \
                        --my_seed $seed --resume "yes"  --load_best_model_at_end  --wandb_mode "dryrun" --infer_path $infer_inp --my_infer_file $infer_out

        infer="span-qa"
        infer_inp="span-qa-inp.tsv"
        infer_out="span-qa-out"
        output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
        echo "** $infer Training SpanDetect(QA_with_charpattern) base_model=$base_model  output_dir=$output_dir"
        python main_span.py --seed $seed --model_name $model_type  --dataset_dir $data \
                        --num_train_epochs $train_epochs --max_seq_len $seq_length --model_mode "roberta_std" \
                        --data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
                        --loss_type "dice" --query_type "question" \
                        --evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
                        --metric_for_best_model "micro_f1" --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --learning_rate "5e-5" \
                        --my_seed $seed --resume "yes"  --load_best_model_at_end  --wandb_mode "dryrun" --infer_path $infer_inp --my_infer_file $infer_out

        infer="span-qa-nocharpattern"
        infer_inp="span-qa-nocharpattern-inp.tsv"
        infer_out="span-qa-nocharpattern-out"
        output_dir="$out_root/$data/$model_type/run-$seed/checkpoints"
        echo "** $infer Training SpanDetect(QA_with_charpattern) base_model=$base_model  output_dir=$output_dir"
        python main_span.py --seed $seed --model_name $model_type  --dataset_dir $data \
                        --num_train_epochs $train_epochs --max_seq_len $seq_length --model_mode "roberta_std" \
                        --data_root $data_root --out_root $out_root --base_model $base_model --output_dir $output_dir  \
                        --loss_type "dice" --query_type "question" \
                        --evaluation_strategy "steps" --eval_steps $eval_step --save_steps $eval_step  --save_total_limit 2 \
                        --metric_for_best_model "micro_f1" --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --learning_rate "5e-5" \
                        --my_seed $seed --resume "yes"  --load_best_model_at_end  --wandb_mode "dryrun" --infer_path $infer_inp --my_infer_file $infer_out

        
done