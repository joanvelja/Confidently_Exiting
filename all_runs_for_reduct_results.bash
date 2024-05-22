wandb login 94c83d220ddc780120eaa22226adf6730f644c6c

srun python run_question_answering.py \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_seq_length 512 \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \

srun python run_question_answering.py \
    --model_name_or_path jvelja/t5-squad \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_seq_length 512 \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \


srun python run_summarization.py \
    --model_name_or_path jvelja/t5-samsum \
    --do_eval \
    --dataset_name samsum \
    --output_dir ./save/samsum_t5_large/ \
    --per_device_eval_batch_size 1 \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \

srun python run_summarization.py \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name samsum \
    --output_dir ./save/samsum_t5_large/ \
    --per_device_eval_batch_size 1 \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \


srun python run_question_answering.py \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_seq_length 512 \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \
    --type_vocab_reduct fixed \

srun python run_question_answering.py \
    --model_name_or_path jvelja/t5-squad \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_seq_length 512 \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \
    --type_vocab_reduct fixed \

srun python run_summarization.py \
    --model_name_or_path jvelja/t5-samsum \
    --do_eval \
    --dataset_name samsum \
    --output_dir ./save/samsum_t5_large/ \
    --per_device_eval_batch_size 1 \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \
    --type_vocab_reduct fixed \

srun python run_summarization.py \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name samsum \
    --output_dir ./save/samsum_t5_large/ \
    --per_device_eval_batch_size 1 \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \
    --type_vocab_reduct fixed \

srun python run_question_answering.py \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_seq_length 512 \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \
    --type_vocab_reduct decaying \

srun python run_question_answering.py \
    --model_name_or_path jvelja/t5-squad \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5_large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_seq_length 512 \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \
    --type_vocab_reduct decaying \

srun python run_summarization.py \
    --model_name_or_path jvelja/t5-samsum \
    --do_eval \
    --dataset_name samsum \
    --output_dir ./save/samsum_t5_large/ \
    --per_device_eval_batch_size 1 \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \
    --type_vocab_reduct decaying \

srun python run_summarization.py \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name samsum \
    --output_dir ./save/samsum_t5_large/ \
    --per_device_eval_batch_size 1 \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_early_exit True \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 14 \
    --exit_conf_type softmax \
    --type_vocab_reduct decaying \