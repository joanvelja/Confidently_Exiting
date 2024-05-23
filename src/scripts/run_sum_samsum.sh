# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
#     run_summarization.py \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --dataset_name samsum \
#     --output_dir ./save/samsum_t5_large/ \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 32 \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --source_prefix "summarize: " \
#     --save_steps 5475 \
#     --learning_rate 1e-4 \
#     --num_train_epochs 20 \

    # FREE
    # --output_hidden_states_decoder True \
    # --intermediate_loss_fn shallowdeep_kd_dyna \
    # --shallow_exit_layer 6 \
    # --distill_layer_alpha 0.5 \
    # --do_layer_transformation False \

    # CALM
    # --output_hidden_states_decoder True \
    # --intermediate_loss_fn weighted_ce \

    # for t5-3b
    # --bf16 \
    # --bf16_full_eval \
    # --use_lora \
    # --lora_rank 64 \
    # --lora_alpha 128 \
    # --lora_target_modules 'q' 'k' 'v' 'o' 'wi' 'wo' \


CUDA_VISIBLE_DEVICES=0 python -m run_summarization \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name samsum \
    --output_dir ./save/jvelja_t5-large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize False \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --use_early_exit True \
    --exit_conf_type softmax \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 17 \
    --max_eval_samples 100 \
    --include_inputs_for_metrics True \
    --use_auth_token True \

    # FREE
    # --use_shallow_deep True \
    # --shallow_exit_layer 6 \
    # --shallow2deep_conf_type softmax \
    # --shallow2deep_conf_threshold 0.9 \
    # --use_adap_threshold True \ # to use adaptive threshold

    # CALM
    # --use_early_exit True \
    # --exit_conf_type softmax \
    # --exit_conf_threshold 0.9 \
    # --exit_min_layer 4 \

    # static-exiting
    # --static_exit_layer 6 \

    # evaluate only performance
    # --deploy_scenario False \
    # --per_device_eval_batch_size 8 \

    # for t5-3b
    # --use_lora \