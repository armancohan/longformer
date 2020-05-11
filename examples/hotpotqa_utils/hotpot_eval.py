""" adapted from the original evaluation script"""
import sys
import re
import string
from collections import Counter, defaultdict
import pickle
import torch
import numpy as np
import logging
import collections
import json
import re
import numpy as np
from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import accuracy_score as accuracy_sklearn
# from .convert_hotpot_to_squad import title_end, title_beg

logger = logging.getLogger()

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def sp_metrics(prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, prec, recall, f1

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def hotpot_evaluate(prediction, gold, allow_partial_prediction=False, hotpot_format=False):
    """ allow_partial_prediction: if part of the evaluation set is going to be evaluated 
        hotpot_format: original hotpot format
    """
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    cnt_missing = 0.0
    for paragraph in gold['data'][0]['paragraphs'] if not hotpot_format else gold:
        dp = paragraph['qas'][0] if not hotpot_format else paragraph
        cur_id = dp['id'] if not hotpot_format else dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            can_eval_joint = False
            cnt_missing += 1
        else:
            answer = dp['answers'][0]['text'] if not hotpot_format else dp['answer']
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], answer)
        if cur_id not in prediction['sp']:
            can_eval_joint = False
        else:
            supporting_facts = paragraph['supporting_facts_raw'] if not hotpot_format else paragraph['supporting_facts']
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], supporting_facts)

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold['data'][0]['paragraphs']) if not hotpot_format else len(gold)
    if allow_partial_prediction:
        print(f'{N-cnt_missing} of {N} gold data have answers, (evaluating only on present answers)')
        N -= cnt_missing
    for k in metrics.keys():
        metrics[k] /= N
    return metrics

def eval_file(prediction_file, gold_file, hotpot_format=True):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
    metrics= hotpot_evaluate(prediction, gold, hotpot_format=hotpot_format)
    print(metrics)


def extract_supporting_sentence(example, sent_labels):
    """ hotpotqa results should be in the following format ['paragraph title': 0] where 0 is an example sentence index
    showing the sentence index in that paragraph, this will decode predictions in that format """
    sents = example.context_text.split('</s>')[:-1]  # last four tokens are `| noans yes no`
    sent_to_paragraph = {}
    sent_to_paragraph_index = {}
    paragraph = ''
    for i, sent in enumerate(sents):
        if title_beg in sent:
            paragraph = re.search(rf'{title_beg} (([\w\W]+)) {title_end}', sent).group(1)
            paragraph_index = 0
        sent_to_paragraph[i] = paragraph
        sent_to_paragraph_index[i] = paragraph_index
        paragraph_index += 1
    results = []
    for i, e in enumerate(sent_labels):
        if e != 1:
            continue
        try:
            result_par = sent_to_paragraph[i]
            result_sent = sent_to_paragraph_index[i]
            results.append([result_par, result_sent])
        except KeyError:
            continue
    return results


def compute_supporting_sentence_metrics(
    all_examples,
    all_features,
    all_results,
    max_sents
):
    metric_results = defaultdict(list)
    results = {}
    for example, feature, result in zip(all_examples, all_features, all_results):
        y_true = np.array([e for e in feature.supporting_facts if e != -1])
        y_pred = torch.tensor(result.sent_logits).max(dim=1)[1].data.numpy()
        if len(y_true) != len(y_pred):
            logger.warning(f'prediction and gold lengths are not equal: y_true: {len(y_true)}, y_pred: {len(y_pred)}')
            y_pred = y_pred[:len(y_true)]
        f1 = f1_score_sklearn(y_true, y_pred)
        acc = accuracy_sklearn(y_true, y_pred)
        metric_results['f1'].append(f1)
        metric_results['accuracy'].append(acc)
        supporting_sents_results = extract_supporting_sentence(example, y_pred)
        results[example.qas_id] = supporting_sents_results
    return results, metric_results

if __name__ == '__main__':
    eval_file(sys.argv[1], sys.argv[2])
