<!--
# Not all FLOPs are created equally: leveraging confidence in intermediate representations to maximize efficiency subject to calibration error
-->

# Optimizing Predictions: Vocabulary Reduction and Contrastive Decoding in LLMs

### Karim Abdel Sadek, Gabriele Desimini, Matteo Nulli, Joan Velja, Jort Vincenti


[**Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding**](https://arxiv.org/abs/2310.05424)       
[Sangmin Bae](https://www.raymin0223.com)$^\*$,
[Jongwoo Ko](https://sites.google.com/view/jongwooko)$^\*$,
[Hwanjun Song](https://songhwanjun.github.io)$^\dagger$,
[Se-Young Yun](https://fbsqkd.github.io)$^\dagger$<br/>
\* equal contribution $&nbsp$ $\dagger$ corresponding author

- **Early-Exiting** dynamically allocates computation paths based on the complexity of generation for each token.
- Conventional framework failed to show actual speedup due to the large number of exit points and state copying mechanism.
- We propose **FREE**, consists of (1) shallow-deep module, (2) synchronized parallel decoding, and (3) adaptive threshold estimator.
- In contrast to conventional approaches, FREE achieved larger inference speedup on extensive generation tasks.

## Requirements
Install the necessary packages with: 
```
$ pip install -r requirements.txt
```

## Experiments
We experimented with 4 summarization tasks, 1 question answering task, and 1 machine translation task.     
Please see the [scripts](scripts/) and run shell files to train or evaluate on each dataset.    
```bash
$ bash run_[TASK_NAME]_[DATASET_NAME].sh
```

### Methods

In addition to the previous implementation we added a few paramters for our task. Please refer [additional_args](src/util/additional_args.py) for more details.   

#### Plotting Softmax: 
- `--plotting_logits False`: If set to True this will plot the confidence, f1, and boxplot of the paper.
- `--final_flops False`: If set to True this will showcase the amount of flops calculated during confidence estimation
- `type_vocab_reduct [str]`: Can be either fixed, decaying, or adaptive and will prune the vocabulary matrix.


### Checkpoints


We share finetuned checkpoints in [google drive](https://drive.google.com/drive/folders/1covxgJtIbFgH_xI-sXIuashX2zsY42w_?usp=share_link).     
Note that you must download `tokenizer.json` for each model individually from HuggingFace to run it without errors. (refer to Issue [#3](https://github.com/raymin0223/fast_robust_early_exit/issues/3))


## Contact
- Sangmin Bae: bsmn0223@kaist.ac.kr
- Jongwoo Ko: jongwoo.ko@kaist.ac.kr