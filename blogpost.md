# Confidence is All you need 

### Karim Abdel Sadek, Gabriele Desimini, Matteo Nulli, Joan Velja, Jort Vincenti
---

## Introduction


Recent advancements in Large Language Models (LLMs) have significantly bolstered performance across various Natural Language Processing (NLP) tasks such as text generation, question answering, and machine translation *inter alia*[[7]](#1), [[8]](#1), [[9]](#1), [[10]](#1). These models, leveraging deep transformer architectures 11 \citep{vaswani2017attention}, generate language tokens autoregressively and provide a flexible framework across tasks with unified natural language input and output [[12]](#1), [[13]](#1), [[14]](#1). Efforts at improving the capabilities of these models have revolved around scaling the number of parameters and data \citep{kaplan2020scaling, hoffmann2022training}. In this context, previous literature show the power of \textit{scaling laws}, which suggest that increasing the size of these models typically yields logarithmic improvements in performance across diverse linguistic benchmarks. These findings indicate that larger models are not only better at generalizing across more complex tasks but also show enhanced \textit{efficiency in learning} from a vast array of data, providing more precise and contextually appropriate outputs. However, the substantial computational load presents a practical challenge during inference, particularly in resource-constrained applications or when serving these models at large scales. To address this issue, early-exiting mechanisms \citep{schwartz2020right,zhu2021leebert, simoulin2021many, bae2023fast} have been proposed, which allow LLMs to dynamically decide the number of layers to use based on the complexity of the input, thus reducing the inference time without significantly compromising performance. This approach, extensively adopted in other contexts in Machine Learning, is crucial because while scaling model architectures is beneficial during training, the same extensive computation may not be necessary at inference time for every input, especially for simpler tasks \citep{geva-etal-2021-transformer, geva-etal-2022-transformer}. By enabling intermediate layer decoding, early exiting offers a promising solution to balance computational efficiency and model accuracy, ensuring that LLMs remain practical and effective in diverse application scenarios. To address this limitation, we propose two-fold improvements over the existing early-exit literature, in a way that enables both efficient and qualitative intermediate layer decoding. 


In this work, we analyze the early exiting paradigm for LLMs, and present a principled method for increasing model efficiency while remaining confident in the quality of the resulting predictions.
Our analysis covers challenges associated with the early-exiting framework. First, we study a phenomena of non-finetuned LMs, where the confidence at early layer is deemed to be high, but where accuracy is not satisfactory. This gives us grounds to experiment some heuristics for the minimum exit layer. Second, we provide a comparative analysis of finetuned versus non-finetuned LMs on a per-task basis, where we provide evidence of further attainable speedups under the finetuned regime.
Specifically, drawing from \cite{schuster2022confident}, we develop a method for calibrating local, per-token, exit decisions such that global, sequence-level constraints —as determined by lexical or semantic sequence-level metrics like ROUGE score— are provably maintained with arbitrarily high probability (e.g., 95\%). Moreover, in order to offset the we leverage within-model \textit{contrastive decoding} \citep{li2023contrastive}, attaining a pareto improvement over runtime and performance.


## Related Works
There have been a large number of studies introducing different Early-Exiting frameworks to adress the increase inference time of Large Language Models \cite{schwartz2020right, simoulin2021many, bae2023fast, zhu2021leebert}. Early-Exiting is based on the intuition that each token needs distinct amounts of compute during generation. Not all tokens necessitate the same amount of compute to be generated, as such many methods have been implemented to achieve this. Some use a routing prediction method \cite{liu2021faster}, others employ an early-exit classifier \cite{schuster2021consistent}, while most of the work is done through softmax-based confidence measures [[6]](#1).

SOFTMAX PART


Introduced by [[4]](#1) *Contrastive Decoding* is a technique used to reduce unwanted behaviours in Large Language Models such as repetition and incoherence. The method is employing two models, a smaller one called amateur and a larger one, called expert. They both perform auto-regressive text generation on the same data, and the final predicted token is selected based on the outputs difference between the predictions of the expert and amateur. While this method is innovative, employing two LLMs is highly inefficient, both in terms of space and compute. Alternative methods have been proposed, which employ the contrastive decoding scheme, without the necessity of using two large models. An example of such work is the idea of Auto-Contrastive Decoding [[2]](#1) . The authors show how contrasting outputs of different layers within the same model can benefit text generation outputs. The study proves that predictions of shallow layers, which are often overlooked, can help those of deeper ones to attain better results. Other studies have adapted this technique to different tasks such as reducing hallucination in LLMs [[3]](#1). 

Our proposed confidence measures connect [[6]](#1) with [[2]](#1) and [[3]](#1). We do so by speeding up the softmax operation of [[6]](#1) and apply auto-contrastive decoding and Jensen-Shannon Divergence to early-exit framework together with the speedup framework of section [[Softmax Speedup]](#1).


##  <a name="Methodology">Methodology</a> 

### <a name="Softmax Speedup">Softmax Speedup </a>

### <a name="Contrastive Decoding ">Contrastive Decoding </a>



#### Weighted contrastive decoding
We call the first `Weighted contrastive decoding`. This method is an adapted version of Auto-contrastive Decoding of [[2]](#1).

 
#### Jensen-Shannon Divergence contrastive decoding
The `Jensen-Shannon Divergence (JSD) contrastive decoding` is inspired by [[3]](#1).



### Speed-up applied to Contrastive Decoding

## Results

## Conclusions



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
