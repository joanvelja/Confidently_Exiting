# Not all FLOPs are created equally: leveraging confidence in intermediate representations to maximize efficiency subject to calibration error 

### Karim Abdel Sadek, Gabriele Desimini, Matteo Nulli, Joan Velja, Jort Vincenti
---

## Introduction



Recent advancements in Large Language Models (LLMs) have significantly bolstered performance across various Natural Language Processing (NLP) tasks such as text generation, question answering, and machine translation (Devlin et al., 2019; Brown et al., 2020; Rae et al., 2021; Smith et al., 2022; Chowdhery et al., 2023, *inter alia*). These models, leveraging deep transformer architectures (Vaswani et al., 2017), generate language tokens autoregressively and provide a flexible framework across tasks with unified natural language input and output (Radford et al., 2019; Raffel et al., 2020; Touvron et al., 2023). Efforts at improving the capabilities of these models have revolved around scaling the number of parameters and data (Kaplan et al., 2020; Hoffmann et al., 2022). In this context, previous literature shows the power of scaling laws, which suggest that increasing the size of these models typically yields logarithmic improvements in performance across diverse linguistic benchmarks. These findings indicate that larger models are not only better at generalizing across more complex tasks, but also show improved efficiency in learning from a vast array of data, providing more precise and contextually appropriate outputs. However, the substantial computational load presents a practical challenge during inference, particularly in resource-constrained applications or when serving these models at large scales. To address this issue, early-exiting techniques (Teerapittayanon et al., 2016; Schwartz et al., 2020; Zhu, 2021; Simoulin & Crabbé, 2021; Bae et al., 2022) have been proposed, which allow LLMs to dynamically decide the number of layers to use based on the complexity of the input, thus reducing the inference time without significantly compromising performance. This approach, extensively adopted in other contexts in Machine Learning, is crucial because while scaling model architectures is beneficial during training, the same extensive computation may not be necessary at inference time for every input, especially for simpler tasks (Geva et al., 2021; 2022). By enabling intermediate layer decoding, early exiting offers a promising solution to balance computational efficiency and model accuracy, ensuring that LLMs remain practical and effective in diverse application scenarios. To address this limitation, we propose twofold improvements over the existing early-exit literature, in a way that enables both efficient and qualitative intermediate layer decoding.

Concretely, we assume we are given a set $S := \{P_i\}_{i=1}^n \in P^n$ of independent and identically distributed _(i.i.d.)_ prompts, each belonging to different tasks (summarization, machine translation, question-answering). 

We allow $P_{test}$ to be an i.i.d. test prompt to our LLM, where $Y_{early} := LLM_{early}(P_{test})$ and $Y_{full} := LLM_{full}(P_{test})$ are the early exiting and standard outputs of our LLM, respectively. In order for $Y_{early}$ to be satisfied with $Y_{full}$, we require it to be both textually consistent - i.e., to be sensical, accurate to a constant - and to have a smaller decoding runtime with respect to $Y_{full}$. To formalize our framework, we thus provide the following constraint:  

```math
P \left( E[D(Y_{early}, Y_{full})] \leq \delta \right) \geq 1 - \epsilon
```

Constraining on textual consistency with the original $Y_{full}$, however, can be cumbersome for most tasks: in summarization, for example, multiple generations may be acceptable; or in question-answering, writing a date in different formats may cause inconsistencies with the ground truth. Within Eq. (1), the goal of our work is to find the most computationally efficient $Y_{early}$ that is consistent with $Y_{full}$ while still maintaining our desired performance guarantee.

In this work, we analyze the early exiting paradigm for LLMs, and present a principled method for increasing model efficiency while remaining confident in the quality of the resulting predictions. Our preliminary analysis covers challenges associated with the early-exiting framework. First, we study a phenomena of non-finetuned LMs, where the confidence at early layer is deemed to be high, but where accuracy is not satisfactory, thus resulting in poor calibration (Mielke et al., 2022; Band et al., 2024). This gives us grounds to experiment some heuristics for the minimum exit layer. Second, we provide a comparative analysis of finetuned versus non-finetuned LMs on a per-task basis, where we provide evidence of further attainable speedups under the finetuned regime. Specifically, drawing from Schuster et al. (2022), we develop a method for calibrating local, per-token, exit decisions such that global, sequence-level constraints — as determined by lexical or semantic sequence-level metrics like ROUGE score — are provably maintained with arbitrarily high probability (e.g., 95%). Moreover, in order to offset the decrease in performance, we leverage within-model contrastive decoding (Li et al., 2023), attaining a pareto improvement over runtime and performance.

Finally, we empirically validate our method on different NLP generation tasks, including text summarization and question answering. Our experiments demonstrate that, combining the two aforementioned approaches, we are able to actually attain the desiderata, that are gains in FLOPs efficiency and improvements in **FINISH HERE**

## Related Works
While the semantic nature of Natural Language is rich, some parts of a sentence, often lack variance. In such cases the amount of layers the model has to go through to understand the right token is relatively low. Following this intuition, there have been a large number of studies introducing different Early-Exiting frameworks aimed at exploiting the reduction in compute for selected tokens, thus reducing the always increasing inference time of Large Language Models (Teerapittayanon et al., 2016; Schwartz et al., 2020; Simoulin & Crabbé, 2021; Zhu, 2021; Bae et al., 2023; Geva et al., 2021; 2022). Many methods have been implemented to achieve this, some use a routing prediction method (Liu et al., 2021), others employing an early-exit classifier (Schuster et al., 2021). However, most of the work is done through softmax-based confidence measures (Schuster et al., 2022). Here, the challenge of early-exiting is addressed by introducing a framework that dynamically allocates computational resources per input and generation time-step. Local exit decisions are calibrated per token but maintaining sequence-level constraints. This ensures computational efficiency without excessive performance degradation. The study also validates their claims both through theoretical analysis and experimentation on diverse text generation tasks and datasets. Differently, Zhu (2021) introduces a new learning scheme. LeeBERT implements a bi-level optimization problem and a cross-level optimization algorithm. This method allows each exit to learn from others and adjusts the weights of different loss terms. Experiments on the GLUE benchmark demonstrate its effectiveness compared to other state-of-the-art early-exit methods. Bae et al. (2023) is instead addressing the early-exiting challenge by introducing the Fast and Robust Early Exiting framework (FREE). This framework’s main contribution is the introduction of a shallow-deep module determining one computational path to be chosen and therefore whether the full model layers need to be used or only part of them is sufficient. Alternatively, Wang et al. (2024) proposes a class-based early-exiting strategy. This method foresees the use of intermediate layer features to exclude part of the tokens, allowing later layers to focus on a reduced subset of potential tokens. Experimental results prove the effectiveness in dynamic neural network inference by reducing computational costs while maintaining high performance.

This research paper is based on the Early-Exit Framework introduced by Schuster et al. (2022). We propose an alternative, which reduces the number of Floating Point Operations per second (FLOPs), making it faster and computationally more efficient. Additionally, we combine it with a textitContrastive Decoding framework (Li et al. (2023)). Contrastive Decoding is a technique used to reduce unwanted behaviours in Large Language Models such as repetition and incoherence. The method is employing two models, a smaller one called amateur and a larger one, called expert. They both perform auto-regressive text generation on the same data, and the final predicted token is selected based on the output divergence between the prediction of the expert and amateur. While this method is innovative, employing two LLMs is highly inefficient, both in terms of space and compute. Alternative methods have been proposed, which employ the contrastive decoding scheme, without the necessity of using two large models. An example of such work is the idea of Auto-Contrastive Decoding (Gera
et al., 2023). The authors show how contrasting outputs of different layers within the same model can benefit text generation outputs. The study proves that predictions of shallow layers, which are of shallow layers, which are often overlooked, can help those of deeper ones to attain better results. Other studies have adapted this technique to different tasks such as reducing hallucination in LLMs (Chuang et al., 2024). Our proposed contrastive decoding techniques are based on both Gera et al. (2023) and Chuang et al. (2024) and adapted to the aforementioned early-exiting framework of Schuster et al. (2022).

# 3 Preliminaries

## 3.1 Transformer Architecture

The Transformer network, introduced by Vaswani et al. (2017), is structured into $L$ layers, each comprising two distinct sublayers: the Multi-Head Attention (MHA) layer and the Feed-Forward Network (FFN) layer. Within this framework, updates to the residual stream for a subsequent prediction are carried out via the following recursive formula:

```math
h^{\ell} = \text{Transformer}^{\ell}(h^{\ell-1})
```

where $\ell$ represents each layer from 1 to $L$, and $h^{\ell+1}$ denotes the output of the embedding layer $W_E$. The embedding layer $W_E \in \mathbb{R}^{d_{vocab} \times d_{model}}$, transforms the tokens $y_{1:t}$ having size $d_{vocab}$, into dense vector representations sized $d_{model}$.

After processing through the $L$-th layer, the prediction for the next token, $\hat{x}_{t+1}$, is produced by:

```math
p(\hat{x}_{t+1} \mid x_{<t+1}) = \text{Softmax}(W_L h_L^{\ell})
```


$$


where $W_L \in \mathbb{R}^{d_{model} \times d_{vocab}}$ is the linear classifier of block $L$ responsible for mapping back the output of the FNN at that block from $d_{model}$ to $d_{vocab}$.

Our approach incorporates an early-exiting strategy, wherein the generation of the next token can occur at any layer $\ell$ if the computed confidence score $c_\ell$ exceeds a specified threshold $\tau$.

When an early exit is triggered at layer $\ell$, it necessitates updating the key and value pairs in subsequent layers to ensure proper attention mechanisms for future tokens. To efficiently manage this, a state copying technique is employed, where the hidden states from the early-exited layer $h_\ell^{\ell+1}$ are duplicated across subsequent layers ($h_{i+1} = h_{i+\ell}$ for every $i$ from $\ell + 1$ to $L$). This process maintains computational efficiency and model performance, even in compact - for today's standards - model configurations like T5 models.

## 3.2 Models, Datasets and Implementation Preliminaries → Title is a Placeholder

# 4 Methodology → Karim, Joan

## 4.1 Early Exiting via the Softmax Approach

Our first approach aims to improve a limitation of the Softmax response method introduced by Schuster et al. (2022). We denote the final output of layer $\ell$ as:

```math
v^{\ell} = \text{Softmax}(W_h h_\ell^{\ell})
```

The so-called confidence measure is computed as the difference between the top two values of the probits vector $v$, at each layer $\ell$. We denote this measure as $c_{\ell+1}$. Let us define an early-exit threshold $\tau_{\ell+1}^{\ell}$ at each layer. If our confidence measure exceeds the early exit-threshold,

```math
c_{\ell+1} > \tau_{\ell+1}^{\ell}
```

the model exits early, providing us with the prediction for the next token computed at that layer. Otherwise, it continues by going into the next Transformer block. However, the matrix multiplication inside $\text{Softmax}(W_h h_\ell^{\ell})$ is computationally expensive, especially when iterated over multiple layers. The exact number of computations for the matrix multiplication above corresponds to

##  <a name="Methodology">Methodology</a> 

### <a name="Early Exiting Via the Softmax approach">Early Exiting Via the Softmax approach</a>
Our first approach aims to improve the speedup of the Softmax response approach introduced by [[6]](#1). The aforementioned confidence measure computes the difference between the top two values of Softmax $(\textbf{W}_i d_t^i)$, at each layer $i$. We denote this measure as $c_t^i$. If $c_t^i \geq \lambda_t^i$, the model exists early, providing us the current prediction computed at that layer.

However, the multiplication inside Softmax, i.e. $\textbf{W}_i d_t^i$ can be deemed as computationally expensive. Note that $\textbf{W}_i \in \mathbb{R}^{30.000 \times dim_d}$, where $30.000$ is our vocabulary size, and  $dim_d$ is equal to the size of the last hidden representation $d_t^i$ of our model. We note and argue that most of these computations are redundant, and potentially not necessary. In Figures below, we show the boxplots for the rank of the final predicted token at each layer, across not  fine-tuned and fine-tuned models, for two different datasets.
The main message these images convey to us is that the final predicted token is often already highly ranked from the first few layers of our model. This behavior is more explicit in Figure [boxplot2](src/plots/boxplot_topk_rank_evalsquad_google-t5_t5-large.png) and [boxplot4](src/plots/boxplot_topk_rank_evalsquad_jvelja_t5-squad.png), where we use fine-tuned models for our downstream task.

![boxplot1](src/plots/boxplot_top1_rank_evalsamsum_google-t5_t5-large.png)
![boxplot2](src/plots/boxplot_topk_rank_evalsquad_google-t5_t5-large.png)
![boxplot3](src/plots/boxplot_top1_rank_evalsamsum_jvelja_t5-samsum.png)
![boxplot4](src/plots/boxplot_topk_rank_evalsquad_jvelja_t5-squad.png)

On the other hand, confidence alone can be a deceiving measure. LLMs can be too confident in the first layers, causing the model to exit prematurely. Our desiderata is for the model to be confident at the same when its prediction has a high accuracy. However, we find that this is not always the case.
In Figure [confidence](src/plots/conf_acc_temp.png), we see the accuracy and the confidence across each layer. The model in the first layers presents an anomalous high confidence, while its performance is poor. Early exiting only based on the Softmax response would result in highly poor performances. We decide to set a *minimum exit layer* parameter $j$, which forces the model to consider exiting only after this layer. Note that this parameter is highly dependent on the model and dataset one experiments on. For fine-tuned models for example, one expects this parameter to be smaller.

![confidence](src/plots/conf_acc_temp.png)


Motivated by these findings, we introduce two additional modifications to the Softmax response approach.

#### Softmax response via fixed pruning
After the minimum early exit layer $j$, we prune $W_j$, retaining its top-k tokens in the new vocabulary matrix. We define the new pruned matrix as $\tilde{W}_{j+i} \in \mathbb{R}^{k \times dim_d}$, for $i = 1, \dots, L-j$ and $k \ll 30.000$. Hence, we prune our matrix at layer $j+1$, and keep the size fixed to $k$ for all subsequent layers 

#### Softmax response via decaying pruning
As one can note from Figures 
[boxplot1](src/plots/boxplot_top1_rank_evalsamsum_google-t5_t5-large.png),
[boxplot2](src/plots/boxplot_topk_rank_evalsquad_google-t5_t5-large.png),
[boxplot3](src/plots/boxplot_top1_rank_evalsamsum_jvelja_t5-samsum.png)
and [boxplot4](src/plots/boxplot_topk_rank_evalsquad_jvelja_t5-squad.png), the rank of the predicted token smoothly decreases across layers, especially for non-fine-tuned models. Again, we prune the $W_j$ matrix, given a minimum early exit layer j. We retain its top k-tokens, obtaining the new pruned vocabulary matrix $\tilde{W}_{j+i} \in \mathbb{R}^{k \times dim_d}. Now, instead of keeping the reduced matrix size fixed, we further prune it after every layer. Given the vocabulary matrix $W_{j+i}$ at layer $j+i$ of size $k_1$, we prune for layer $j+i+1$ it to a reduced matrix of size $k_2$, where

```math
k_2 = \max\left(k^*, \left\lfloor \frac{k1}{1 + \frac{k1 - k^*}{k^*} \cdot \frac{j+i}{\text{num\_layers}}} \right\rfloor \right)
```

$k^*$ here indicates the minimum size our vocabulary matrix $\textbf{W}_{j+i+1}$ can reach. 

The benefit of this approach is obvious, and justified. Our predicted token is often in the top-k ones, with a high value of $k$. Due to this, pruning the vocabulary matrix allows us to reduce the amount of computations we have to compute at each layer, while discarding only irrelevant tokens. While we trade-off some performance, this further speeds up the runtime of our model, allowing us to obtain notable efficiency gains.


### <a name="Contrastive Decoding ">Contrastive Decoding </a>

The second approach is based on results from [[4]](#1). The aforementioned work proposes *contrastive decoding* as a search-based decoding method, inspired by the fact that the failures of larger LMs, e.g., repetition, incoherence, are even more prevalent in smaller LMs, and that this difference signals which texts should be preferred. The original implementation involves the use of two models in parallel, returning the difference between the likelihood under a large LM - called the *expert* - and a small LM - called the *amateur*. We speculate the potential improvement of our original notion of confidence with this approach. We apply contrastive decoding by overcoming a clear limitation of the original work, that is parallel inference among the amateur and the expert models. For this reason, we use previous layers predictions as a proxy of the amateur model, thus removing the requirement of parallel decoding.


#### Weighted contrastive decoding
We call the first `Weighted contrastive decoding`. This method is an adapted version of Auto-contrastive Decoding of [[2]](#1).

#### Jensen-Shannon Divergence contrastive decoding
The `Jensen-Shannon Divergence (JSD) contrastive decoding` is inspired by [[3]](#1) but applied to the early-exiting framework.



### Speed-up applied to Contrastive Decoding

## Results
We evaluate the encoder-decoder t5-large model [[22]](#1) on five different datasets and three different downstream tasks:

- Stanford Question Answering Dataset (SQuAD) with over 16k of annotated data [[23]](#1).
- SamSum a human-annotated dataset for abstractive Summarization [[24]](#1).
- CNN/DailyMail, over 300k news articles from CNN and DailyMail [[25]](#1).
- Multi-News a long-context multi-document Summarization dataset with news articles [[26]](#1).
- IWSLT 2017 [[27]](#1) a benchmark for Machine Translation for multiple language (English and German) directions.

We also compare the performance and effects of our proposed methods between the pre-trained only version [t5-large](https://huggingface.co/google-t5/t5-large), [long-t5-tglobal-base](https://huggingface.co/google/long-t5-tglobal-base) and the fine-tuned version of t5-large on each training dataset [t5-squad](https://huggingface.co/jvelja/t5-squad),[t5-samsum](https://huggingface.co/jvelja/t5-samsum), [t5-cndm](https://huggingface.co/jvelja/t5-cnndm), [t5-multinews](https://huggingface.co/jvelja/t5-multinews), [t5-bigpatent](https://huggingface.co/jvelja/t5-bigpatent) and [t5-IWSLT](https://huggingface.co/jvelja/t5-iwslt).

More to come at the final deadline.

### Softmax Results

This table displays the type of softmax exiting (either decaying, fixed, or none) in relation to performance metrics such as runtime, F1 score, Rouge, or BLEU. A clear pattern emerges: when the reduction of the vocabulary size is changed from none to either decaying or fixed, performance in terms of F1, Rouge, or BLEU decreases. However, the overall runtime also decreases, illustrating a trade-off where reducing the vocabulary size results in faster processing times but lower performance.

An interesting pattern appears for the fine-tuned models: if the average block exit is similar across all reduction types, the "None" reduction type yields the best performance across all metrics. This is because vocabulary reduction benefits only if the model computes the logits through most of the layers. When this occurs, it increases the overall runtime.


| Model | Dataset | Reduction Type | Block Average Exit | Runtime | Samples per second | f1, rougeL or bleu |
|:----------------------------|:--------------|:--------------------|-----------------:|---------------:|--------------------------:|----------:|
| t5-large | squad | decaying | 15 | 6.7521 | 1.481 | 0  |
| t5-large | squad | fixed | 15.05 | 2.6543 | 3.768 | 0 |
| t5-large | squad | None | 19.5294 | 5.2598 | 1.901 | 90 |
| t5-squad | squad | decaying | 14.0612 | 4.3845 | 2.281 | 80 |
| t5-squad | squad | fixed | 14.566 | 6.9129 | 1.447 | 80 |
| t5-squad | squad | None | 14.566 | 4.4011 | 2.272 | 100 |
| t5-samsum | samsum | decaying | 14.9615 | 11.5839 | 0.863 | 2.5 |
| t5-samsum | samsum | fixed | 15 | 161.101 | 0.062 | 0 |
| t5-samsum | samsum | None | 19.6388 | 21.3617 | 0.468 | 36.0191 |
| t5-large | samsum | decaying | 15 | 18.6154 | 0.537 | 5.3799 |
| t5-large | samsum | fixed | 15 | 166.882 | 0.06 | 0  |
| t5-large | samsum | None | 21.3266 | 43.0316 | 0.232 | 18.3291 |
| long-t5-tglobal-base | multi_news | decaying | 9.54358 | 66.1225 | 0.151 | 3.3908 |
| long-t5-tglobal-base | multi_news | fixed | 9.17888 | 97.5776 | 0.102 | 13.0121 |
| long-t5-tglobal-base | multi_news | None | 9.03257 | 90.7552 | 0.11 | 16.0328 |
| t5-large | iwslt2017 | decaying | 14.9921 | 10.69 | 0.935 | 0.165 |
| t5-large | iwslt2017 | fixed | 15 | 72.6153 | 0.138| 0 |
| t5-large | iwslt2017 | None | 19.8665 | 21.4535 | 0.466 | 0.6646 |
| t5-large | cnn_dailymail | decaying | 15 | 7.5626 | 1.322  | 0.8696 |
| t5-large | cnn_dailymail | fixed | 15 | 65.6162 | 0.152 |  0  |
| gt5-large | cnn_dailymail | None | 20.1062 | 35.9859 | 0.278 | 23.1428 |
| t5-cnndm | cnn_dailymail | decaying | 14.92 | 7.3143 | 1.367 | 2.046 |
| t5-cnndm | cnn_dailymail | fixed | 15 | 20.9629 | 0.477 | 0 |
| t5-cnndm | cnn_dailymail | None | 18.1947 | 47.876 | 0.209 | 24.9601 |
| long-t5-tglobal-base | big_patent | decaying | 9.82461 | 55.6787 | 0.18 | 4.6623 |
| long-t5-tglobal-base | big_patent | fixed | 9.3767 | 154.648 | 0.065  | 10.1653 |
| long-t5-tglobal-base | big_patent | None | 9.12285 | 158.71 | 0.063 | 22.6895 |


---

**Note:** Due to Snellius being down, we had to make these runs locally on 100 samples, which can result in inaccuracies in the results. Additionally, some models could not be fitted into memory and were therefore ignored.

## Conclusions
Conclusions will be written at the final deadline. 


## References
<a id="1">[1]</a>
Bae, Sangmin, Jongwoo Ko, Hwanjun Song, and Se-Young Yun. "Fast and robust early-exiting framework for autoregressive language models with synchronized parallel decoding." arXiv preprint arXiv:2310.05424 (2023).

<a id="1">[2]</a>
Gera, Ariel, Roni Friedman, Ofir Arviv, Chulaka Gunasekara, Benjamin Sznajder, Noam Slonim, and Eyal Shnarch. "The benefits of bad advice: Autocontrastive decoding across model layers." arXiv preprint arXiv:2305.01628 (2023).

<a id="1">[3]</a>
Chuang, Yung-Sung, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. "Dola: Decoding by contrasting layers improves factuality in large language models." arXiv preprint arXiv:2309.03883 (2023).

<a id="1">[4]</a>
Li, Xiang Lisa, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. "Contrastive decoding: Open-ended text generation as optimization." arXiv preprint arXiv:2210.15097 (2022).

<a id="1">[5]</a>
Schuster, Tal, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. "Confident adaptive language modeling." Advances in Neural Information Processing Systems 35 (2022): 17456-17472.

<a id="1">[6]</a>
Schuster, Tal, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. "Confident adaptive language modeling." Advances in Neural Information Processing Systems 35 (2022): 17456-17472.

<a id="1">[7]</a>
Brown, Tom, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.

<a id="1">[8]</a>
Rae, Jack W., Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides et al. "Scaling language models: Methods, analysis & insights from training gopher." arXiv preprint arXiv:2112.11446 (2021).

<a id="1">[9]</a>
Smith, Shaden, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu et al. "Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model." arXiv preprint arXiv:2201.11990 (2022).


<a id="1">[10]</a>
Chowdhery, Aakanksha, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham et al. "Palm: Scaling language modeling with pathways." Journal of Machine Learning Research 24, no. 240 (2023): 1-113.

<a id="1">[11]</a>
Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." Advances in neural information processing systems 30 (2017).

<a id="1">[12]</a>
Radford, Alec, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. "Language models are unsupervised multitask learners." OpenAI blog 1, no. 8 (2019): 9.

<a id="1">[13]</a>
Raffel, Colin, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. "Exploring the limits of transfer learning with a unified text-to-text transformer." Journal of machine learning research 21, no. 140 (2020): 1-67.

<a id="1">[14]</a>
Touvron, Hugo, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière et al. "Llama: Open and efficient foundation language models." arXiv preprint arXiv:2302.13971 (2023).

<a id="1">[15]</a>
Kaplan, Jared, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. "Scaling laws for neural language models." arXiv preprint arXiv:2001.08361 (2020).

<a id="1">[16]</a>
Hoffmann, Jordan, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas et al. "Training compute-optimal large language models." arXiv preprint arXiv:2203.15556 (2022).

<a id="1">[17]</a>
Schwartz, Roy, Gabriel Stanovsky, Swabha Swayamdipta, Jesse Dodge, and Noah A. Smith. "The right tool for the job: Matching model and instance complexities." arXiv preprint arXiv:2004.07453 (2020).

<a id="1">[18]</a>
Zhu, Wei. "LeeBERT: Learned early exit for BERT with cross-level optimization." In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 2968-2980. 2021.

<a id="1">[19]</a>
Simoulin, Antoine and Benoît Crabbé. “How Many Layers and Why? An Analysis of the Model Depth in Transformers.” Annual Meeting of the Association for Computational Linguistics (2021).

<a id="1">[20]</a>
Geva, Mor, Roei Schuster, Jonathan Berant, and Omer Levy. "Transformer feed-forward layers are key-value memories." arXiv preprint arXiv:2012.14913 (2020).

<a id="1">[21]</a>
Geva, Mor, Avi Caciularu, Kevin Ro Wang, and Yoav Goldberg. "Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space." arXiv preprint arXiv:2203.14680 (2022).

<a id="1">[22]</a>
Raffel, Colin, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. "Exploring the limits of transfer learning with a unified text-to-text transformer." Journal of machine learning research 21, no. 140 (2020): 1-67.

<a id="1">[23]</a>
Rajpurkar, Pranav, Jian Zhang, Konstantin Lopyrev, and Percy Liang. "Squad: 100,000+ questions for machine comprehension of text." arXiv preprint arXiv:1606.05250 (2016).

<a id="1">[24]</a>
Gliwa, Bogdan, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. "SAMSum corpus: A human-annotated dialogue dataset for abstractive summarization." arXiv preprint arXiv:1911.12237 (2019).

<a id="1">[25]</a>
See, Abigail, Peter J. Liu, and Christopher D. Manning. "Get to the point: Summarization with pointer-generator networks." arXiv preprint arXiv:1704.04368 (2017).

<a id="1">[26]</a>
Fabbri, Alexander R., Irene Li, Tianwei She, Suyi Li, and Dragomir R. Radev. "Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model." arXiv preprint arXiv:1906.01749 (2019).

<a id="1">[27]</a>
Cettolo, Mauro, Marcello Federico, Luisa Bentivogli, Niehues Jan, Stüker Sebastian, Sudoh Katsuitho, Yoshino Koichiro and Federmann Christian. “Overview of the IWSLT 2017 Evaluation Campaign.” International Workshop on Spoken Language Translation (2017).

<a id="1">[28]</a>
Zhu, Wei. “LeeBERT: Learned Early Exit for BERT with cross-level optimization.” Annual Meeting of the Association for Computational Linguistics (2021).

<a id="1">[29]</a>
Wang, Jingcun, Bing Li, and Grace Li Zhang. "Early Classification for Dynamic Inference of Neural Networks." arXiv preprint arXiv:2309.13443 (2023).

<a id="1">[30]</a>
Liu, Yijin, Fandong Meng, Jie Zhou, Yufeng Chen, and Jinan Xu. "Faster depth-adaptive transformers." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 15, pp. 13424-13432. 2021.

<a id="1">[31]</a>
Schuster, Tal, Adam Fisch, Tommi Jaakkola, and Regina Barzilay. "Consistent accelerated inference via confident adaptive transformers." arXiv preprint arXiv:2104.08803 (2021).

<a id="1">[32]</a>
Schwartz, Roy, Gabriel Stanovsky, Swabha Swayamdipta, Jesse Dodge, and Noah A. Smith. "The right tool for the job: Matching model and instance complexities." arXiv preprint arXiv:2004.07453 (2020).
