#!/bin/bash
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part0_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part1_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part2_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part3_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part4_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part5_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part6_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part7_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part8_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
CUDA_VISIBLE_DEVICES=0,4,5,1 python main_qa.py --config config/onto_final_part9_sd42/config_roberta_qa4_querytype4_span_char_pattern3_subtext_dim50.json
