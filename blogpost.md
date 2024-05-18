# Confidence is All you need 

### Karim Abdel Sadek, Gabriele Desimini, Matteo Nulli, Joan Velja, Jort Vincenti
---

## Introduction

In this work we aim to provide an extensive analysis and a new framework on Early-Exiting in Large Language Models. We expand upon [[1]](#1) by adapting [[2]](#1) and [[3]](#1) to an early-exiting process and propose a novel procedure which attains faster generation time, by retaining almost all performance when compared to full model without early-exiting.

## Related Works

Recent advancement in Large Language Models capabilities have come at a computational time and price. Inference time has severely increased, causing these models to be slower then ever. In recent times different Early-Exiting frameworks [[1]](#1), [[5]](#1), (add other papers) have been proposed to address this issue... 

(`Matteo`: dabi maybe you can revisit this part and add information about softmax)

(`Matteo`: From here forward I'll introduce contrastive decoding)

Introduced by [[4]](#1) *Contrastive Decoding* is a technique used to reduce unwanted behaviours in Large Language Models such as repetition and incoherence. The method is employing two models, a smaller one called amateur and a larger one, called expert. They both perform autoregressive text generation on the same data, and the final predicted token is selected based on the outputs difference between the predictions of the expert and amateur. While this method is innovative and performant, employing two LLMs is highly inefficient, both in terms of space and compute.Alternative methods have been proposed to counter the necessity of using two large models. An example of such work is the idea of Auto-Contrastive Decoding, introduced in [[2]](#1). Here, the authors show how contrasting outputs of different layers within the same model can benefit text generation outputs. The study proves that predictions of shallow layers, which are often overlooked, can help those of deeper ones to attain better resutls. (? Extend with information from [[2]](#1) ?)
The two proposed confidence measures in Section [[Contrastive Decoding]](#1) are based on this paper, but are both adapted to an early-exiting framework and later extended in [[Speed-up applied to Contrastive Decoding]](#1).


##  <a name="Methodology">Methodology</a> 

### Softmax speed-up

### <a name="Contrastive Decoding ">Contrastive Decoding </a>

#### Weighted contrastive decoding
We call the first `Weighted contrastive decoding`. This method is an adapted version of Auto-contrastive Decoding of [[2]](#1).

 
#### Jensen-Shannon Divergence contrastive decoding
The `Jensen-Shannon Divergence (JSD) contrastive decoding` is inspired by [[3]](#1).



### <a name="Speed-up applied to Contrastive Decoding">Speed-up applied to Contrastive Decoding</a>

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