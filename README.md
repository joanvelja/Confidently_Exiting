<!--
# Not all FLOPs are created equally: leveraging confidence in intermediate representations to maximize efficiency subject to calibration error
-->

# Optimizing Predictions: Vocabulary Reduction and Contrastive Decoding in LLMs

This repo is cloned from the code-base <a href="https://github.com/raymin0223/fast_robust_early_exit" target="_blank" rel="noopener noreferrer">  fast_robust_early_exit</a> with the original [paper](https://arxiv.org/abs/2310.05424). We futher extend their works by adding prunning options on the logits before softmax evaluation, and implemented contrastive decoding. Our discussion and findings can be found in the [blogpost](blogpost.md) file.


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

In addition to the previous implementation parameters we added new paramters for our task. Please refer [additional_args](src/util/additional_args.py) for more details.   

#### Plotting Softmax: 
- `type_vocab_reduct [str]`: Can be either fixed, decaying, or adaptive and will prune the vocabulary matrix.
- `--plotting_logits False`: If set to True this will plot the confidence, f1, and boxplots (Figure 2,3, and 4 of the [blogpost](blogpost.md)).
- `--final_flops False`: If set to True this will showcase the amount of flops calculated during confidence estimation (Figure 6 and 7 of the [blogpost](blogpost.md)).



### Checkpoints


To run the finetuned models they are available at [jveila](https://huggingface.co/jvelja) on the HuggingFace platform.


## Contact
- Karim Abdel Sadek
- Gabriele Desimini
- Matteo Nulli
- Joan Velja
- Jort Vincenti