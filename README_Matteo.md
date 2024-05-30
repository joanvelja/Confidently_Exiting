<!--
# Not all FLOPs are created equally: leveraging confidence in intermediate representations to maximize efficiency subject to calibration error
-->

# Optimizing Predictions: Vocabulary Reduction and Contrastive Decoding in LLMs

This repository is cloned from the code-base <a href="https://github.com/raymin0223/fast_robust_early_exit" target="_blank" rel="noopener noreferrer">  Fast_Robust_Early_Exit</a> (their [paper](https://arxiv.org/abs/2310.05424)) . We further extend their work by implementing our two proposed approaches: Softmax Exiting with reduced voabulary size, and implemented Contrastive Decoding. Our discussion and findings can be found in our [blogpost](blogpost.md) file.


## Requirements
In order to set up the environment for reproducing our experiments, install the necessary packages with: 
```
$ pip install -r requirements.txt
```
Or via the environment file:
```
conda env create --name environment_name -f environment.yml
```

The codebase handles automatically model and dataset downloading. Beware of this when running the code for the first time! 


## Evaluation
We perform evaluation experiments on one summarization and one question answering task. 

To reproduce the experiments you can follow the guide below.

#### Softmax Pruning
Please see the [scripts/softmax_experiments](src/scripts/softmax_experiments) shell files to reproduce the Softmax experiments on each dataset.    
```bash
sh jobname.run > jobname.out
```

#### Contrastive Decoding
Here we explain how to reproduce the experiments from section `Contrastive Decoding` of our [blogpost](blogpost.md). 

The experiments of Figures [Figure 8a](./blogpost_images/plots/squadexit.png), [Figure 8b](./blogpost_images/plots/squadf1.png), [Figure 9a](./blogpost_images/plots/sam_avg.png), [Figure 9b](./blogpost_images/plots/samsum_intermediate.png) and Table 1 are carried out across 100 samples. To reproduce these results it is enough to run the files of [scripts/contrastive_decoding_experiments/SQuAD](src/scripts/contrastive_decoding_experiments/SQuAD) and [scripts/contrastive_decoding_experiments/SamSum](src/scripts/contrastive_decoding_experiments/SamSum) by adding an extra parameter namely:

- `--max_eval_samples 100`
  
Similarly, [Figure 10b](./blogpost_images/plots/squad_flops.png),  [Figure 11b](./blogpost_images/plots/sam_flops.png) are performed over 100 samples with the additional need of the `count_flops` parameter

- `--count_flops True`
  
Differently, the results of the last plots [Figure 10a](./blogpost_images/plots/squad_f1.png) and [Figure 11a](./blogpost_images/plots/rougesamsam.png) are made by running the .job files of [scripts/contrastive_decoding_experiments/SQuAD](src/scripts/contrastive_decoding_experiments/SQuAD) and [scripts/contrastive_decoding_experiments/SamSum](src/scripts/contrastive_decoding_experiments/SamSum) without any additional change. 
An example for running Jansen-Shannon Divergence contrastive confidence with adaptive pruning is

```bash
srun python run_question_answering.py \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5-large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_seq_length 512 \
    --use_early_exit True \
    --exit_conf_type JSD_contrastive_confidence \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 19 \
    --include_inputs_for_metrics False \
    --use_auth_token True \
    --type_vocab_reduct adaptive \
```

Additionally, the actual plots are done with the `plots_mn.ipynb` file in (folder you put the plots folder in.) by manually inserting the numbers we obtain from the runs. 



### Parameters Explanation

In addition to the parameters previously implemented, we have introduced new ones specific to our tasks. For further details, please refer to the [additional_args](src/util/additional_args.py) documentation. For convenience, we will also highlight the essential parameters from the previous implementation that are utilized in our current setup.

#### Essential Parameters:

The bash files to run the evals look as follows:

```
srun python run_question_answering.py \
    --model_name_or_path google-t5/t5-large \
    --do_eval \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --output_dir ./save/squad_t5-large/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_seq_length 512 \
    --use_early_exit True \
    --exit_conf_type softmax \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 16 \
    --include_inputs_for_metrics True \
    --max_eval_samples 100 \
    --use_auth_token True \
```

##### Method agnostic parameters
- `-m`: the file responsible for the task. The structure of it is `run_$TASK`. Possible choices: `question_answering`, `summarization`.
- `--model_name_or_path`: the model to be used for the task. Possible choices: `google-t5/t5-large`, `jvelja/t5-squad`, `jvelja/t5-samsum`.
- `--do_eval` True: this should be always True for evals.
- `--deploy_scenario` True: this should be always True to use deploying_[MODEL_NAME].py for our implementation.
- `--use_early_exit` True: use conventional early-exiting framework.
- `--exit_conf_threshold` [float]: threshold value to decide whether to exit or not. Our experiments were made with 0.9.
- `--exit_min_layer` [int]: the minimum number of layers to forward to decide the exiting. 


##### Softmax
- `--exit_conf_type softmax`: set the confidence measure to softmax values
- `--type_vocab_reduct [str]`: Can be either fixed, decaying, or adaptive. This will prune the vocabulary matrix.
- `--plotting_logits False`: if set to True this will plot the confidence, f1, and boxplots (Figure 2,3, and 4 of the [blogpost](blogpost.md)).
- `--final_flops False`: if set to True this will showcase the amount of flops calculated during confidence estimation (Figure 6 and 7 of the [blogpost](blogpost.md)).

##### Contrastive Decoding
- `--exit_conf_type [str]`: Can now also be set to <i>contrastive_decoding</i>, <i>reweight_contrastive_decoding</i>, or <i>JSD_contrastive_confidence</i>.
- `--type_vocab_reduct [str]`: Can be either fixed, decaying, or adaptive. This will prune the vocabulary matrix. This parameter is needed to combine <i>reweight_contrastive_decoding</i>, or <i>JSD_contrastive_confidence</i> with the pruning method.

Sample task-specific bash files can be found in the `src/scripts` directory. 



### W&B logging

To enable wandb logging of results you can follow the standaard procedure explained in [wandb login infos](https://docs.wandb.ai/ref/cli/wandb-login) and uncomment the following lines of code   
and set the statement to "false"

`os.environ["WANDB_DISABLED"] = "true" ---> os.environ["WANDB_DISABLED"] = "false"`

This, together with the usual `wandb.init()` will save every evaluation metric to into your wandb project.
This line of code can be found within [run_question_answering](src/run_question_answering.py) / [run_summarization](src/run_summarization.py).


### Model Checkpoints

The non-finetuned and finetuned models are available at  [google](https://huggingface.co/google-t5) and [jvelja](https://huggingface.co/jvelja) respectively on HuggingFace. 

## Contact
- Karim Abdel Sadek: karim.abdel.sadek@student.uva.nl
- Gabriele Desimini: gabriele.desimini@student.uva.nl
- Matteo Nulli: matteo.nulli@student.uva.nl
- Joan Velja: joan.velja@student.uva.nl
- Jort Vincenti: jort.vincenti@student.uva.nl
