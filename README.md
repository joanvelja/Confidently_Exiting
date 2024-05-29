<!--
# Not all FLOPs are created equally: leveraging confidence in intermediate representations to maximize efficiency subject to calibration error
-->

# Optimizing Predictions: Vocabulary Reduction and Contrastive Decoding in LLMs

This repository is cloned from the code-base <a href="https://github.com/raymin0223/fast_robust_early_exit" target="_blank" rel="noopener noreferrer">  Fast_Robust_Early_Exit</a>( [paper](https://arxiv.org/abs/2310.05424)) . We further extend their work by implementing our two proposed approaches: softmax exiting with reduced voabulary size, and implemented contrastive decoding. Our discussion and findings can be found in the [blogpost](blogpost.md) file.


## Requirements
Install the necessary packages with: 
```
$ pip install -r requirements.txt
```
Or via the environment file:
```
conda env create --name environment_name -f environment.yml
```

## Experiments
We experimented with 1 summarization and 1 question answering task. 
Please see the [scripts/softmax_experiments](src/scripts/softmax_experiments) shell files to reproduce the softmax experiments on each dataset.    
```bash
sh jobname.run > jobname.out
```

### Methods

In addition to the parameters previously implemented, we have introduced new ones specific to our tasks. For further details, please refer to the [additional_args](src/util/additional_args.py)  documentation. For convenience, we will also highlight the essential parameters from the previous implementation that are utilized in our current setup.

#### Essential Parameters:
- `--use_early_exit True`: use conventional early-exiting framework 
- `--exit_min_layer [int]`: the minimum number of layers to forward to decide the exiting
- `--exit_conf_threshold [float]`: threshold value to decide whether to exit or not

#### Softmax: 
- `--exit_conf_type softmax`: set the confidence measure to softmax values
- `--type_vocab_reduct [str]`: Can be either fixed, decaying, or adaptive. This will prune the vocabulary matrix.
- `--plotting_logits False`: If set to True this will plot the confidence, f1, and boxplots (Figure 2,3, and 4 of the [blogpost](blogpost.md)).
- `--final_flops False`: If set to True this will showcase the amount of flops calculated during confidence estimation (Figure 6 and 7 of the [blogpost](blogpost.md)).

#### Contrastive Decoding:
- `--exit_conf_type [str]`: Can now also be set to <i>contrastive_decoding </i>, <i> reweight_contrastive_decoding </i>, or <i>JSD_contrastive_confidence</i>.


### Model Checkpoints


The finetuned models are available at [jveila](https://huggingface.co/jvelja) on HuggingFace.

## Contact
- Karim Abdel Sadek: karim.abdel.sadek@student.uva.nl
- Gabriele Desimini: gabriele.desimini@student.uva.nl
- Matteo Nulli: matteo.nulli@student.uva.nl
- Joan Velja: joan.velja@student.uva.nl
- Jort Vincenti: jort.vincenti@student.uva.nl