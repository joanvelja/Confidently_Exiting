import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig
from util.JSD import JSD

def confidence_measure(
    method: str,
    lm_logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: nn.Linear = None,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    layer_am: int = None,
    alpha: float = None,
    return_jsds=False,
    return_logits=False,
):
    assert lm_logits is not None

    output = {}

    if method == 'softmax':
        probs = torch.softmax(lm_logits, dim=-1)
        top_2 = torch.topk(probs, dim=-1, k=2)[0]
        output['conf'] = (top_2[..., 0] - top_2[..., 1]).squeeze()
    
    elif method == 'meta':
        assert hidden_states is not None and classifier is not None
        preds = classifier(hidden_states)
        probs = torch.softmax(preds, dim=-1)
        output['conf'] = probs[..., 1].squeeze()

    elif method in ['contrastive_decoding', 'reweight_contrastive_decoding', 'JSD_contrastive_confidence']:
        probits_exp = torch.softmax(lm_logits, dim=-1).squeeze_()
        prev_probits[layer_exp] = probits_exp
        max_probs_exp = torch.max(probits_exp)

        if method == 'JSD_contrastive_confidence':
            jsd = JSD()
            mask = probits_exp >= alpha * max_probs_exp
            jsds = {layer: jsd(probits_exp[mask], prev_probits[layer][mask]) / (layer_exp - layer) for layer in np.arange(stop=layer_exp, start=2)}
            max_jsd_layer = max(jsds, key=jsds.get)
            probits_am = prev_probits[max_jsd_layer]
            s = torch.zeros_like(probits_exp)
            contrast = torch.log(probits_exp[mask]) - torch.log(probits_am[mask])
            s[mask] = torch.softmax(contrast, dim=-1).mul_(torch.sum(probits_exp))
            s[~mask] = probits_exp[~mask]
            top_2 = torch.topk(s, dim=-1, k=2)[0]
            output['conf'] = (top_2[..., 0] - top_2[..., 1]).squeeze()
            if return_jsds:
                output['jsds'] = jsds
            if return_logits:
                output['logits'] = s
        
        elif method == 'contrastive_decoding':
            if layer_am not in prev_probits:
                raise ValueError("Chosen layer has not been computed yet")
            probits_am = prev_probits[layer_am]
            s = torch.zeros_like(probits_exp)
            mask = probits_exp >= alpha * max_probs_exp
            s[mask] = torch.softmax(torch.log(probits_exp[mask]) - torch.log(probits_am[mask]), dim=-1)
            top_2 = torch.topk(s, dim=-1, k=2)[0]
            output['conf'] = (top_2[..., 0] - top_2[..., 1]).squeeze()

        elif method == 'reweight_contrastive_decoding':
            if layer_am not in prev_probits:
                raise ValueError("Chosen layer has not been computed yet")
            probits_am = prev_probits[layer_am]
            s = torch.zeros_like(probits_exp)
            mask = probits_exp >= alpha * max_probs_exp
            contrast = torch.softmax(torch.log(probits_exp[mask]) - torch.log(probits_am[mask]), dim=-1).mul_(torch.sum(probits_exp[mask]))
            s[mask] = contrast
            top_2 = torch.topk(s, dim=-1, k=2)[0]
            output['conf'] = (top_2[..., 0] - top_2[..., 1]).squeeze()

    else:
        raise ValueError('Invalid confidence measure: {}'.format(method))

    return output

def get_skip_mask(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: nn.Linear = None,
    config: AutoConfig = None,
    pos_time: int = 1,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    layer_am: int = None,
    alpha: float = 0.1,
    adapt_threshold: float = None,
    return_conf=False,
    return_jsds=False,
    return_logits=False,
):

    assert config.exit_conf_type is not None or config.shallow2deep_conf_type is not None

    if config.exit_conf_type is not None:
        key = config.exit_conf_type
        if config.exit_position_temp is not None:
            correct_by_pos = lambda i: config.exit_conf_threshold * np.exp(
                - config.exit_position_temp * i / config.max_answer_length
            ) / 10 + 9 * config.exit_conf_threshold / 10
            threshold = correct_by_pos(pos_time)
        else:
            threshold = config.exit_conf_threshold
    elif config.shallow2deep_conf_type is not None:
        key = config.shallow2deep_conf_type
        threshold = config.shallow2deep_conf_threshold if adapt_threshold is None else adapt_threshold

    if key == 'JSD_contrastive_confidence':
        output = confidence_measure(
            method=key,
            lm_logits=logits,
            layer_exp=layer_exp,
            prev_probits=prev_probits,
            alpha=alpha,
            return_jsds=return_jsds,
            return_logits=return_logits,
        )
    else:
        output = confidence_measure(
            method=key,
            lm_logits=logits,
            hidden_states=hidden_states,
            classifier=classifier,
            layer_exp=layer_exp,
            prev_probits=prev_probits,
            layer_am=layer_am,
            alpha=alpha,
        )

    conf = output.get('conf')
    mask = torch.where(conf <= threshold, 0., 1.).bool()

    result = {'mask': mask.item()}
    if return_jsds and 'jsds' in output:
        result['jsds'] = output['jsds']
    if return_conf:
        result['conf'] = conf.item()
    if return_logits and 'logits' in output:
        result['logits'] = output['logits']

    return result