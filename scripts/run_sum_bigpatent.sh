# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
#     run_summarization.py \
#     --model_name_or_path google/long-t5-tglobal-base \
#     --do_train \
#     --do_eval \
#     --dataset_name big_patent \
#     --dataset_config_name e \
#     --output_dir ./save/bigpatent_longt5_base/ \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 16 \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --source_prefix "summarize: " \
#     --save_steps 2583 \
#     --learning_rate 1e-4 \
#     --num_train_epochs 3 \
#     --max_source_length 2048 \
#     --max_target_length 512 \

    # FREE
    # --output_hidden_states_decoder True \
    # --intermediate_loss_fn shallowdeep_kd_dyna \
    # --shallow_exit_layer 6 \
    # --distill_layer_alpha 0.5 \
    # --do_layer_transformation False \

    # CALM
    # --output_hidden_states_decoder True \
    # --intermediate_loss_fn weighted_ce \


# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
#     run_summarization.py \
#     --model_name_or_path ./save/bigpatent_longt5_base/ \
#     --do_eval \
#     --dataset_name big_patent \
#     --dataset_config_name e \
#     --output_dir ./save/bigpatent_longt5_base/ \
#     --per_device_eval_batch_size 1 \
#     --deploy_scenario True \
#     --use_synchronize True \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --source_prefix "summarize: " \
#     --max_source_length 2048 \
#     --max_target_length 512 \

CUDA_VISIBLE_DEVICES=0 python -m run_summarization \
    --model_name_or_path jvelja/t5-bigpatent \
    --do_eval \
    --dataset_name big_patent \
    --dataset_config_name e \
    --output_dir ./save/big_patent_t5-large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize False \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --max_source_length 2048 \
    --max_target_length 512 \
    --use_early_exit True \
    --exit_conf_type JSD_contrastive_confidence \
    --exit_conf_threshold 1.1 \
    --exit_min_layer 2 \
    --max_eval_samples 10 \
    --include_inputs_for_metrics True \

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
