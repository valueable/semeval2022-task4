#! /bin/bash
task=1
model_name=roberta-large-task1-new-6
device=7
CUDA_VISIBLE_DEVICES=$device nohup python3 -u train.py \
	--train_set ../data/train_task${task}_whole.txt \
	--dev_set ../data/dev_task${task}_whole.txt \
	--ptr_dir ../pretrain/roberta_large_itpt \
	--eval_step 500 \
	--save_dir ../checkpoints/${model_name} \
	--train_batch_size 16 \
	--learning_rate 1.5e-5 \
	--max_seq_length 256 \
	--rdrop_coef 0 \
	--dropout 0 \
	--warmup_ratio 0 \
	--weight_decay 0 \
	--epochs 50 \
	--focal_loss 1 \
	--multilabel_celoss 1 \
	--accu_gradient 1 \
	--threshold 0.44 \
	--weightsample 0 \
	--kw 1 \
	--use_wandb 1 \
	--task ${task} \
	--model_name ${model_name} \
	>../logs/train_${model_name}.log 2>&1 &