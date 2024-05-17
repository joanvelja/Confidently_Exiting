# Confidence is All you need 

### Karim Abdel Sadek, Gabriele Desimini, Matteo Nulli, Joan Velja, Jort Vincenti
---

## Introduction

In this work we aim to provide an extensive analysis and a new framework on Early-Exiting in Large Language Models. We expand upon [[1]](#1) by adapting [[2]](#1) and [[3]](#1) to an early-exiting process and propose a novel procedure which attains faster generation time, by retaining almost all performance when compared to full model without early-exiting.

## Related Works

Recent advancement in Large Language Models capabilities have come at a computational time and price. Inference time has severely increased, causing these models to be slower then ever. In recent times different Early-Exiting frameworks [[5]](#1), (another) have been proposed to address this issue... 
Contrastive decoding [[4]](#1) is used to reduce unwanted behaviours in Large Language Models such as repetition and incoherence. The method is employing two models, a smaller one called amateur and a much larger one, called expert, for autoregressive generation. While this method is innovative and performant, employing two LLMs is highly inefficient. 
The two proposed confidence measures are based on the intuition that the same method can be reproduced between layers of the same model. Deeper layers of the model can benefit from simpler information present in shallower ones. This idea, introduced in [[2]](#1) is the basis for both our proposed confidence measures. 

## Methodology

### Softmax speed-up

### Contrastive Decoding as alternative confidence measure

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