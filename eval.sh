#! /bin/bash
'''
model_name=roberta-large-itpt-task1
task=1
python3 -u predict.py \
	--params_path checkpoints/${model_name}/model_best_step/model_state.pt \
	--model_path pretrain/roberta_large_itpt \
	--batch_size 128 \
	--input_file data/dev_task1.txt \
	--max_seq_length 256 \
	--task ${task}
'''
model_name=roberta-large-task2-threshold0.44-bs8
task=2
python3 -u predict.py \
	--params_path checkpoints/${model_name}/model_best_step/model_state.pt \
	--model_path model/roberta-large \
	--batch_size 128 \
	--seed 42 \
	--max_seq_length 500 \
	--threshold 0.44 \
	--input_file data/dev_task2.txt \
	--task ${task}