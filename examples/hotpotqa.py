import argparse
import itertools
import json
import math
import os
import pathlib
import random
import re
from collections import defaultdict
from itertools import chain, combinations
from typing import Iterable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.overrides.data_parallel import (
    LightningDataParallel, LightningDistributedDataParallel)
from torch._six import container_abcs, int_classes, string_classes
from torch.nn.modules import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

try:
    from apex import amp
except ImportError:
    from torch.cuda import amp

from hotpotqa_utils.hotpot_eval import exact_match_score
from hotpotqa_utils.hotpot_eval import f1_score as hotpot_f1_score
from hotpotqa_utils.hotpot_eval import hotpot_evaluate, sp_metrics
from hotpotqa_utils.hotpot_prep import (SENT_MARKER_END, TITLE_END,
                                        get_roberta_tokenizer)
from hotpotqa_utils.hotpot_utils import get_final_text
from longformer import sliding_chunks
from longformer.longformer import Longformer
from longformer.sliding_chunks import pad_to_window_size

FIXED_SEED = 9090

dev_json_cache = None


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class GeLU(nn.Module):

    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(torch.nn.functional, "gelu"):
            return torch.nn.functional.gelu(x.float()).type_as(x)
        else:
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def map_ans_type(answer):
    if answer == 'yes':
        q_type = 0
    elif answer == 'no':
        q_type = 1
    else:
        q_type = 2
    return q_type

int_to_answer_type = {0: 'yes', 1: 'no', 2: 'span'}


def calc_f1(y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)
    return f1


def _pad_and_get_mask(tokens, max_len, padding_item):
    """ pad tokens to max_len and get corresponding mask """
    mask = [1] * len(tokens)
    padding_len = max_len - len(tokens)
    tokens.extend([padding_item] * padding_len)
    mask.extend([0] * padding_len)
    return tokens, mask


def custom_collate(batch):
    # 0  'q_id': instance['_id'],
    # 1  'q_tokens': question_tokens,
    # 2  'doc_tokens': all_doc_tokens,
    # 3  'start_pos': start_positions,
    # 4  'end_pos': end_positions,
    # 5  'answer_str': instance.get('answer'),
    # 6  'sent_labels': sentence_labels,
    # 7  'num_sents': sent_cnt,
    # 8  'num_pars': len(paragraph_sents),
    # 9  'pars': paragraph_sents,
    # 10 'sent_idx': list(zip(sent_indices_start, sent_indices_end)),
    # 11 'par_idx': list(zip(par_indices_start, par_indices_end)),
    # 12 'sent_to_par_idx': sent_to_par_index
    # 13 'par_labels'
    # 14 'q_type'
    # 15 'token_to_orig_map'
    # 16 'orig_doc_tokens'
    # 17 'entity_attention'    

    transposed = zip(*batch)
    lst = []
    q_lens, doc_lens, sent_lens = [], [], []
    for i, samples in enumerate(list(transposed)):
        if i == 1 or i == 2:
            # roberta.decoder.dictionary.pad() is 1
            sequences_padded = nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=1)
            lengths = torch.LongTensor([len(x) for x in samples])
            lst.extend([sequences_padded])
            if i == 1:
                q_lens = lengths
            else:
                doc_lens = lengths
        elif i == 6 or i == 13: # for sentence labels
            # we'll use ignore_index == -1 in sentence loss
            sequences_padded = nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=-1)
            lengths = torch.LongTensor([len(x) for x in samples])
            lst.extend([sequences_padded])
            if i == 6:
                sent_lens = lengths
            if i == 13:
                par_lens = lengths
        elif i in {3, 4, 7, 8, 14, 17}:
            out = None
            elem = samples[0]
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in samples])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            lst.append(torch.stack(samples, 0, out=out))
        elif i in {0, 5, 9, 10, 11, 12, 15, 16}:
            lst.append(samples)
        else:
            raise RuntimeError('invalid batch element')
    lst.extend([q_lens, doc_lens, sent_lens, par_lens])
    return lst


class HotpotDataset(Dataset):

    def __init__(self, file_path, max_seq_len, num_samples=None, split='train'):
        self.data = []
        with open(file_path) as fin:
            for i, line in enumerate(tqdm(fin, desc=f'loading input file {file_path.split("/")[-1]}', unit_scale=1)):
                self.data.append(json.loads(line))
                if num_samples and len(self.data) > num_samples:
                    break
        self.max_seq_len = max_seq_len
        self._tokenizer = get_roberta_tokenizer()

        # A mapping from qid to an int, which can be synched across gpus using `torch.distributed`
        if 'train' not in split:  # not for the training set
            self.val_qid_string_to_int_map =  \
                {
                    entry["q_id"]: index
                    for index, entry in enumerate(self.data)
                }
        else:
            self.val_qid_string_to_int_map = None
        self.is_train = 'train' in split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._convert_to_tensors(self.data[idx])

    def _convert_to_tensors(self, instance):
            # 0  'q_id': instance['_id'],
    # 1  'q_tokens': question_tokens,
    # 2  'doc_tokens': all_doc_tokens,
    # 3  'start_pos': start_positions,
    # 4  'end_pos': end_positions,
    # 5  'answer_str': instance.get('answer'),
    # 6  'sent_labels': sentence_labels,
    # 7  'num_sents': sent_cnt,
    # 8  'num_pars': len(paragraph_sents),
    # 9  'pars': paragraph_sents,
    # 10 'sent_idx': list(zip(sent_indices_start, sent_indices_end)),
    # 11 'par_idx': list(zip(par_indices_start, par_indices_end)),
    # 12 'sent_to_par_idx': sent_to_par_index
    # 13 'entity_attention'

        # list of wordpiece tokenized candidates
        q_token_ids = self._tokenizer.convert_tokens_to_ids(instance['q_tokens'])
        doc_token_ids = self._tokenizer.convert_tokens_to_ids(instance['doc_tokens'])
        q_type = map_ans_type(instance['answer_str']) if instance.get('answer_str') else []

        return (instance['q_id'], 
                torch.tensor(q_token_ids), 
                torch.tensor(doc_token_ids),
                torch.tensor(instance['start_pos']),
                torch.tensor(instance['end_pos']),
                instance['answer_str'],
                torch.tensor(instance['sent_labels']),
                torch.tensor(instance['num_sents']),
                torch.tensor(instance['num_pars']),
                instance['pars'],
                instance['sent_idx'],
                instance['par_idx'],
                instance['sent_to_par_idx'],
                torch.tensor(instance.get('par_labels')),
                torch.tensor(q_type),
                instance.get('token_to_orig_map') if not self.is_train else 0,
	            instance.get('orig_doc_tokens') if not self.is_train else 0,
                torch.tensor(instance.get('entity_attention') or []))


def mlp_classification_head(input_dim, hidden_dim, output_dim, activation='gelu'):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        GeLU() if activation == 'gelu' else nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def get_activations(model, q_ids, doc_ids, max_seq_len, kwargs, extra_attn, symmetric_extra_attention, attention_step, use_segment_ids, sent_token_id, model_type, par_token_id=None, pad_token_id=None, attention_window=None, entity_attention=None):
    q_len = q_ids.shape[1]
    doc_len = doc_ids.shape[1]

    attn_mode = model.encoder.layer[0].attention.self.attention_mode
    if attn_mode in {'tvm', 'sliding_chunks'} and extra_attn:
        # always attend to the canidate_ids
        include_extra_attention_mask = True
    else:
        include_extra_attention_mask = False
    extra_attention_mask = None

    if q_len + doc_len <= max_seq_len:
        token_ids = torch.cat([q_ids, doc_ids], dim=1)
        extra_attention_mask = None
        sentence_mask = token_ids == sent_token_id  # sentence end location
        if par_token_id is not None:
            par_mask = token_ids == par_token_id # paragraph start location
            sentence_mask = par_mask | sentence_mask
        if entity_attention is not None:
            entity_attention = torch.cat([q_ids.new_zeros(q_ids.shape, dtype=bool), entity_attention.bool()], dim=1)
            sentence_mask = entity_attention | sentence_mask
        extra_attention_mask = torch.zeros(token_ids.shape, dtype=torch.bool, device=token_ids.device)
        extra_attention_mask[sentence_mask] = 1  # attend to all sentences
        extra_attention_mask[:, :q_len] = 1  # attend to question tokens

        if model_type == 'longformer':
            token_ids, extra_attention_mask = pad_to_window_size(token_ids, extra_attention_mask, attention_window, pad_token_id)

            features = model(token_ids, attention_mask=extra_attention_mask)[0]

            # remove padding tokens before computing loss and decoding
            padding_len = token_ids[0].eq(pad_token_id).sum()
            if padding_len > 0:
                features = features[:, :-padding_len]
        elif model_type == 'roberta':
            features = model.extract_features(token_ids)[0]
        else:
            raise False
        return [features]

    else:  # sliding window
        all_activations = []
        all_token_ids = []
        all_sentence_masks = []
        available_support_len = max_seq_len - q_len
        for start in range(0, doc_len, available_support_len):
            end = min(start + available_support_len, doc_len)

            token_ids = torch.cat([q_ids, doc_ids[:, start:end]], dim=1)
            sentence_mask = token_ids == sent_token_id  # sentence end location
            if par_token_id is not None:
                par_mask = token_ids == par_token_id # paragraph start location
                sentence_mask = par_mask | sentence_mask
            if entity_attention is not None:
                sentence_mask = entity_attention | sentence_mask
            extra_attention_mask = torch.zeros(token_ids.shape, dtype=torch.bool, device=token_ids.device)
            extra_attention_mask[sentence_mask] = 1  # attend to all sentences
            for i in range(token_ids.shape[0]):
                extra_attention_mask[i, :q_len] = 1

            token_ids, extra_attention_mask = pad_to_window_size(token_ids, extra_attention_mask, attention_window, pad_token_id)

            activations = model(token_ids, attention_mask=extra_attention_mask)[0]

            # remove padding tokens before computing loss and decoding
            padding_len = token_ids[0].eq(pad_token_id).sum()
            if padding_len > 0:
                features = features[:, :-padding_len]
            all_activations.append(activations)

        return all_activations

class HotpotModel(pl.LightningModule):

    def __init__(self, args):
        super(HotpotModel, self).__init__()
        self.args = args
        self.hparams = args

        self.roberta = self.load_roberta()
        embed_dim = self.roberta.embeddings.word_embeddings.weight.shape[1]
        self.answer_score = torch.nn.Linear(embed_dim, 1, bias=False)

        self.signpost_interval = -1  # not used
        self.hparams = args

        self._tokenizer = get_roberta_tokenizer()

        self._extract_features_args = {}

        if hasattr(self.args, 'question_type_classification_head') and args.question_type_classification_head:
            self.qa_type_classifier = mlp_classification_head(embed_dim, embed_dim // 2, 3)
            self.qa_type_loss = CrossEntropyLoss(ignore_index=-1)

# Build a feed-forward network
        if hasattr(self.args, 'multi_layer_classification_heads') and self.args.multi_layer_classification_heads:
            if not hasattr(self.args, 'mlp_qa_head') or args.mlp_qa_head:
                self.qa_outputs = mlp_classification_head(embed_dim, embed_dim, 2)
            else:
                self.qa_outputs = nn.Linear(embed_dim, 2)
            self.sentence_classifier = mlp_classification_head(embed_dim, embed_dim, 2)
            self.paragraph_classifier = mlp_classification_head(embed_dim, embed_dim, 2)
        else:
            self.qa_outputs = nn.Linear(embed_dim, 2)
            self.sentence_classifier = nn.Linear(embed_dim, 2)
            if hasattr(self.args, 'paragraph_loss') and self.args.paragraph_loss:
                self.paragraph_classifier = nn.Linear(embed_dim, 2)
        self.sentence_loss = CrossEntropyLoss(ignore_index=-1)
        self.paragraph_loss = CrossEntropyLoss(ignore_index=-1)
        self.running_metrics = defaultdict(float)
        self.sent_token_ids = self._tokenizer.convert_tokens_to_ids([SENT_MARKER_END])[0]
        self.par_token_id = self._tokenizer.convert_tokens_to_ids([TITLE_END])[0]
        self.val_dataloader_obj = None
        self.test_dataloader_obj = None

        print("== requires_grad ==")
        total_num_param = 0
        trainable_param = 0
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
            total_num_param += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
        print(f'---\n\nTrainable params: {trainable_param:,}')
        print(f'Total params: {total_num_param:,}\n\n---')
        print(args)

    def load_roberta(self):
        if self.args.model_type == 'roberta':
            from fairseq.models.roberta import RobertaModel
            model = RobertaModel.from_pretrained(self.args.model_path, checkpoint_file=self.args.model_filename)
        elif self.args.model_type in ['longformer', 'tvm_roberta']:
            model = Longformer.from_pretrained(self.args.model_path)
            model.resize_token_embeddings(50272)
            for layer in model.encoder.layer:
                layer.attention.self.attention_mode = self.args.attention_mode
                self.args.attention_window = layer.attention.self.attention_window
            # create additional projection matrices in the self-attention layers
            # for the candidates to context attention
            # embed_dim = model.args.encoder_embed_dim
            # has_bias = model.model.decoder.sentence_encoder.layers[0].self_attn.q_proj.bias is not None
            # for layer in model.model.decoder.sentence_encoder.layers:
            #     for key in ['k', 'v', 'q']:
            #         proj_full = torch.nn.Linear(embed_dim, embed_dim, bias=has_bias)
            #         proj_full.weight.data.copy_(getattr(layer.self_attn, key + '_proj').weight.data)
            #         if has_bias:
            #             proj_full.bias.data.copy_(getattr(layer.self_attn, key + '_proj').bias.data)
            #         layer.self_attn.add_module(key + '_proj_full', proj_full)
        else:
            assert False

        print("Loaded model with config:")
        print(model.config)

        for p in model.parameters():
            p.requires_grad_(True)
        model.train()

        return model

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, overrides=None):
        r"""
        Overrides super function to allow passing overrides for loading the model
        """
        from argparse import Namespace
        if map_location is not None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        try:
            ckpt_hparams = checkpoint['hparams']
        except KeyError:
            raise IOError(
                "Checkpoint does not contain hyperparameters. Are your model hyperparameters stored"
                "in self.hparams?"
            )
        if overrides is not None:
            for k in overrides:
                ckpt_hparams[k] = overrides[k]
        hparams = Namespace(**ckpt_hparams)

        # load the state_dict on the model automatically
        model = cls(hparams)
        # model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model

    # def backward(self, use_amp, loss, optimizer):
    #     if self.args.fp16 and self.args.optimizer_type == 'fairseq_optimizer':
    #         # using fairseq optimizer, this is how it is done
    #         optimizer.backward(loss)
    #     else:
    #         if use_amp:
    #             with amp.scale_loss(loss, optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             loss.backward()

    def or_softmax_cross_entropy_loss_one_doc(self, logits, target, ignore_index=-1, dim=-1):
        """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
        assert logits.ndim == 2
        assert target.ndim == 2
        assert logits.size(0) == target.size(0)

        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
        masked_target = target * (1 - target_mask.long())
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = float('-inf')

        # each batch is one example
        gathered_logits = gathered_logits.view(1, -1)
        logits = logits.view(1, -1)

        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
        # denominator = log(sum(exp(logits)))
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

        # compute the loss
        loss = -(log_score - log_norm)

        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
        # remove those from the loss before computing the sum. Use sum instead of mean because
        # it is easier to compute
        return loss[~torch.isinf(loss)].sum()

    def forward(self, q_ids, q_len, doc_ids, sentence_labels=None, start_pos=None, end_pos=None, paragraph_labels=None, q_type_labels=None, entity_attention=None):
        """
        Args:
            q_ids: quesion token ids
            q_len: maximum question tokens length
            doc_ids: document token ids
            sentence_labels: sentence labels for evidence
            start(end)_pos: starting (ending) position for the span
        """

        #roberta, q_ids, doc_ids, max_seq_len, kwargs, do_extra_attention, symmetric_extra_attention, attention_step, use_segment_ids, SENT_TOKEN_ID
        activations = get_activations(
            self.roberta,
            q_ids,
            doc_ids,
            self.args.max_seq_len,
            self._extract_features_args,
            extra_attn=self.args.extra_attn,
            symmetric_extra_attention=self.args.extra_attn, # always symmetric
            attention_step=self.signpost_interval + 1,
            use_segment_ids=self.args.use_segment_ids and self.args.create_new_weight_matrics,
            sent_token_id=self.sent_token_ids,
            model_type=self.args.model_type,
            par_token_id=self.par_token_id if self.args.include_paragraph else None,
            pad_token_id=self._tokenizer.pad_token_id,
            attention_window=self.args.attention_window,
            entity_attention=entity_attention if self.args.include_entities else None)

        if len(activations) > 1:
            # document has been broken into parts
            # reconstruct the original activation
            # activations List[[batch_size, max_seq_len (or shorter), dim]]
            if hasattr(self.args, 'loss_at_different_layers') and self.args.loss_at_different_layers:
                # when passing return_all_hiddens to fairseq it return tensor in shape Time x B x D instead of B x Time x D
                activations = torch.cat([activations[0]] + [e[q_len:,:,:] for e in activations[1:]], dim=0)
                # get activations at layer 20 to attach sentence loss
                sentence_activations = torch.cat([inners[0][21]] + [e[21][q_len:,:,:] for e in inners[1:]], dim=0)
                sentence_activations = sentence_activations.transpose(0, 1)
                activations = activations.transpose(0, 1)
            else:
                activations = torch.cat([activations[0]] + [e[:,q_len:,:] for e in activations[1:]], dim=1)
                assert activations.shape[1] == q_ids.shape[1] + doc_ids.shape[1]
        else:
            activations = activations[0]
            if hasattr(self.args, 'loss_at_different_layers') and self.args.loss_at_different_layers:
                sentence_activations = inners[0][21]
                # when passing return_all_hiddens to fairseq it return tensor in shape Time x B x D instead of B x Time x D
                sentence_activations = sentence_activations.transpose(0, 1)
                activations = activations.transpose(0, 1)

        # pull out </s> tokens
        sentence_mask = doc_ids == self.sent_token_ids
        sentence_mask = torch.cat([q_ids.new_zeros(q_ids.shape).type_as(sentence_mask), sentence_mask], dim=1)

        # shape [num-sent, 2]
        # B, S, D = features.shape # batch size, seq len, dim
        # pull out activations for sequence
        # shape [num-sent, 2]
        # B, S, D = features.shape # batch size, seq len, dim
        if hasattr(self.args, 'loss_at_different_layers') and self.args.loss_at_different_layers:
            prediction_score = self.sentence_classifier(sentence_activations[sentence_mask])
        else:
            prediction_score = self.sentence_classifier(activations[sentence_mask])
        batch_size = prediction_score.new_ones(1) * activations.shape[0]
        predicted_answers = prediction_score.argmax(dim=1)

        if hasattr(self.args, 'question_type_classification_head') and self.args.question_type_classification_head:
            q_type_logits = self.qa_type_classifier(activations[:, 0])
            if q_type_labels.numel() > 0:
                q_type_loss = self.qa_type_loss(q_type_logits, q_type_labels)
                q_type_acc = (q_type_logits.argmax(dim=1) == q_type_labels).int().sum() / torch.tensor(q_type_labels.shape[0], dtype=torch.float32, device=q_type_logits.device)
        else:
            q_type_loss = None
            q_type_logits = None
            q_type_acc = 0.0

        if self.args.include_paragraph and self.args.paragraph_loss:
            par_mask = doc_ids == self.par_token_id
            par_mask = torch.cat([q_ids.new_zeros(q_ids.shape).type_as(par_mask), par_mask], dim=1)
            if hasattr(self.args, 'loss_at_different_layers') and self.args.loss_at_different_layers:
                par_score = self.paragraph_classifier(sentence_activations[par_mask])
            else:
                par_score = self.paragraph_classifier(activations[par_mask])
            par_pred = par_score.argmax(dim=1)
            if paragraph_labels.numel() > 0:
                par_labels_flat = paragraph_labels[paragraph_labels != -1]
                paragraph_loss = self.paragraph_loss(par_score, par_labels_flat)
                par_f1 = calc_f1(par_pred, par_labels_flat)
                par_acc = (par_pred == par_labels_flat).int().sum() / torch.tensor(par_pred.shape[0], dtype=torch.float32, device=par_pred.device)
            else:
                paragraph_loss = par_f1 = par_acc = 0.0

        if sentence_labels.numel() > 0:
            sentence_labels_flat = sentence_labels[sentence_labels != -1]
            sentence_loss = self.sentence_loss(prediction_score, sentence_labels_flat)
            num_correct = (predicted_answers == sentence_labels_flat.squeeze(-1)).int().sum()
            total = predicted_answers.new_ones(1) * predicted_answers.shape[0]
            y_pred = predicted_answers
            y_true = sentence_labels_flat
            f1 = calc_f1(y_pred, y_true)
            sent_accuracy = (y_pred == y_true).int().sum() / torch.tensor(y_pred.shape[0], dtype=torch.float32, device=y_pred.device)
        else:
            loss = prediction_score.new_zeros(1).float()
            num_correct = 0
            sentence_loss = f1 = sent_accuracy = 0.0
        # loss = self.loss(sum_prediction_scores, correct_prediction_index)

        logits = self.qa_outputs(activations)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_pos is not None and end_pos is not None and start_pos.numel() != 0 and end_pos.numel() != 0:
            # recall start and end position labels are offsets in the document
            # we have appended that to the query. So we need to adjust accordingly
            start_positions = start_pos + q_len
            end_positions = end_pos + q_len
            # if len(start_positions.size()) > 1:
            #     start_positions = start_positions.squeeze(-1)
            # if len(end_positions.size()) > 1:
            #     end_positions = end_positions.squeeze(-1)

            if self.args.or_softmax_loss:
                # loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf
                # NOTE: this returns sum of losses, not mean, so loss won't be normalized across different batch sizes
                # but batch size is always 1, so this is not a problem anymore
                start_loss = self.or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1)
                end_loss = self.or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
                start_positions = start_positions[:, 0:1]
                end_positions = end_positions[:, 0:1]
                start_loss = loss_fct(start_logits, start_positions[:, 0])
                end_loss = loss_fct(end_logits, end_positions[:, 0])
                            # top 1 accuracy
            start_accuracy = (start_logits.argmax(dim=1)==start_positions).int().sum() / torch.tensor(start_logits.shape[0], dtype=torch.float32, device=start_logits.device)
            end_accuracy = (end_logits.argmax(dim=1)==end_positions).int().sum() / torch.tensor(end_logits.shape[0], dtype=torch.float32, device=end_logits.device)

            span_loss = (start_loss + end_loss) / 2.0
        else:
            start_accuracy = end_accuracy = 0.0
            span_loss = None

        output = {"span_loss": span_loss, "sent_loss": sentence_loss,
                  "start_logits": start_logits, "end_logits": end_logits,
                  "sent_logits": prediction_score, "par_logits": par_score,
                  "sent_f1": f1, 'sent_accuracy': sent_accuracy,
                  'start_accuracy': start_accuracy, 'end_accuracy': end_accuracy,
                  'q_type_loss': q_type_loss, 'q_type_logits': q_type_logits, 'q_type_acc': q_type_acc}
        if self.args.include_paragraph and self.args.paragraph_loss:
            output['paragraph_loss'] = paragraph_loss
            output['paragraph_f1'] = par_f1
            output['paragraph_acc'] = par_acc
        else:
            output['paragraph_loss'] = 0.0
            output['paragraph_f1'] = 0.0
            output['paragraph_acc'] = 0.0
        return output

    def training_step(self, batch, batch_nb):

        q_id, q_tokens, doc_tokens, start_pos, end_pos, answer_str, sent_labels, num_sents, num_pars, pars, sent_idx, par_idx, sent_to_par_idx, par_labels, q_type_label, token_to_orig_map, orig_doc_tokens, entity_attention, q_lens, doc_lens, sent_lens, par_lens = batch
        n_batch = len(q_tokens)
        output = self.forward(q_tokens, q_lens.max(), doc_tokens, sent_labels, start_pos, end_pos, par_labels, q_type_label, entity_attention=entity_attention)

        if self.args.linear_mixing is None:
            if hasattr(self.args, 'question_type_classification_head') and self.args.question_type_classification_head:
                span_loss = output['span_loss'] + output['q_type_loss']
            else:
                span_loss = output['span_loss']

            # sum the span level loss and support fact loss
            if self.args.dynamic_mixing_ratio:
                # gradually increase mixing ratio to focus more on spans in later
                mixing_ratio = self._num_grad_updates / self.args.total_num_updates
                total_loss = mixing_ratio * span_loss + (1 - mixing_ratio) * output['sent_loss']
                self.args.mixing_ratio = mixing_ratio
            else:
                if self.args.paragraph_loss:
                    total_loss = self.args.mixing_ratio * span_loss + (1 - self.args.mixing_ratio) * ((1- self.args.mixing_ratio_par) * output['sent_loss'] + self.args.mixing_ratio_par * output['paragraph_loss'])
                else:
                    total_loss = self.args.mixing_ratio * span_loss + (1 - self.args.mixing_ratio) * output['sent_loss']
        else:
            mixing = json.loads(self.args.linear_mixing)
            total_loss = mixing[0] * output['span_loss'] + mixing[1] * output['q_type_loss'] +\
                mixing[2] * output['sent_loss'] + mixing[3] * output['paragraph_loss']
        # lr = total_loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        if self.args.optimizer_type == 'fairseq_optimizer':
            lr = total_loss.new_zeros(1) + self.trainer.optimizers[0].optimizer.param_groups[0]['lr']
        else:
            lr = total_loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': total_loss, 'lr': lr, 'train_sent_acc': output['sent_accuracy'],
                            'train_sent_f1': output['sent_f1'], 'train_start_acc': output['start_accuracy'],
                            'train_span_loss': output['span_loss'], 'train_sent_loss': output['sent_loss'], 'mixing_ratio': self.args.mixing_ratio,
                            'num_grad_updates': self._num_grad_updates,
                            'train_par_f1': output.get('paragraph_f1'), 'train_par_acc': output.get('paragraph_acc'),
                            'q_type_acc': output.get('q_type_acc'), 'q_type_loss': output.get('q_type_loss'),
                            'train_paragraph_loss': output.get('paragraph_loss')}

        progress_bar = {'lr': lr,
                        'n_updates': total_loss.new_ones(1) * self._num_grad_updates, 'loss': total_loss}
        result = {'loss': total_loss, 'progress_bar': progress_bar, 'log': tensorboard_logs}
        return result

    def _validation_step(self, batch, batch_number):
        global dev_json_cache
        q_id, q_tokens, doc_tokens, start_pos, end_pos, answer_str, sent_labels, num_sents, num_pars, pars, sent_idx, par_idx, sent_to_par_idx, par_labels, q_type_label, token_to_orig_map, orig_doc_tokens, entity_attention, q_lens, doc_lens, sent_lens, par_lens = batch
        n_batch = len(q_tokens)
        assert n_batch == 1  # doesn't currently support larger batch size
        output = self.forward(q_tokens, q_lens.max(), doc_tokens, sent_labels, start_pos, end_pos, par_labels, q_type_label, entity_attention=entity_attention)

        if self.args.linear_mixing is None:
            if self.args.dynamic_mixing_ratio:
                # gradually increase mixing ratio to focus more on spans in later
                mixing_ratio = self._num_grad_updates / self.args.total_num_updates
                total_loss = mixing_ratio * output['span_loss'] + (1 - mixing_ratio) * output['sent_loss']
            else:
                if hasattr(self.args, 'question_type_classification_head') and \
                        self.args.question_type_classification_head and output['span_loss']:
                    span_loss = output['span_loss'] + output['q_type_loss']
                else:
                    span_loss = output['span_loss']
                if span_loss is None and self.args.mixing_ratio > 0.0:
                    span_loss = output['sent_loss']  # TODO: handle this better, this is only for test time when span_loss becomes none
                if span_loss is None or output['sent_loss'] is None:  # prediction mode
                    total_loss = output['sent_logits'].new_zeros(1)[0]
                else:
                    total_loss = self.args.mixing_ratio * span_loss + (1 - self.args.mixing_ratio) * output['sent_loss']
        else:
            mixing = json.loads(self.args.linear_mixing)
            if output['span_loss'] is None or output['sent_loss'] is None:
                total_loss = torch.tensor(5.0, dtype=output['sent_logits'].dtype, device=output['sent_logits'].device)  # TODO: if labels doesn't exist assign loss of 5.0, handle this better
            else:
                total_loss = mixing[0] * output['span_loss'] + mixing[1] * output['q_type_loss'] +\
                    mixing[2] * output['sent_loss'] + mixing[3] * output['paragraph_loss']
        sent_logits = output['sent_logits']
        start_logits = output['start_logits']
        end_logits = output['end_logits']
        par_logits = output['par_logits']
        type_logits = output['q_type_logits']

        pred_q_type = type_logits.argmax(dim=1) if type_logits is not None else None

        input_ids = torch.cat([q_tokens, doc_tokens], dim=1)
        answers, supporting_facts, _, _ = self.decode(input_ids, start_logits, end_logits, q_lens, sent_logits, par_logits, type_logits, pars, sent_to_par_idx)
        f1s = []
        ems = []
        for answer, gold in zip(answers, answer_str):
            span_f1 = hotpot_f1_score(answer['text'], gold)
            em = exact_match_score(answer['text'], gold)
            f1s.append(span_f1)
            ems.append(em)

        hotpot_f1 = torch.tensor(f1s, dtype=torch.float32, device=q_tokens.device).mean()
        hotpot_em = torch.tensor(ems, dtype=torch.float32, device=q_tokens.device).mean()

        # each batch is one document
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)[0:1]

        tensorboard_logs = {'val_loss': total_loss, 'val_sent_acc': output['sent_accuracy'],
                            'val_sent_f1': output['sent_f1'], 'val_start_acc': output['start_accuracy'],
                            'val_span_loss': output['span_loss'], 'val_sent_loss': output['sent_loss'],
                            'val_span_f1': hotpot_f1, 'hotpot_em': hotpot_em,
                            'val_par_f1': output.get('paragraph_f1'), 'val_par_acc': output.get('paragraph_acc'),
                            'val_par_loss': output.get('paragraph_loss'),
                            'val_q_type_acc': output.get('q_type_acc'), 'val_q_type_loss': output.get('q_type_loss')}
        progress_bar = {'val_sent_f1': output['sent_f1'], 'val_em': hotpot_em, 'val_f1': hotpot_f1}
        result = {'val_loss': total_loss, 'val_em': hotpot_em, 'val_span_f1': hotpot_f1,
                  'val_sent_f1': output['sent_f1'], 'val_span_em': hotpot_em,
                  'answers': answers, 'supporting_facts': supporting_facts,
                  'qid': q_id, 'progress_bar': progress_bar, 'log': tensorboard_logs,
                  'val_par_f1': output.get('paragraph_f1'), 'val_par_acc': output.get('paragraph_acc'),
                  'val_par_loss': output.get('paragraph_loss'),
                  'pred_q_type': pred_q_type, 'q_type_labels': q_type_label}
        return result


    def decode(self, input_ids, start_logits, end_logits, q_len, sent_logits, par_logits, type_logits,
                pars, sent_to_par_idx, simple_sentence_decode=True, fancy_span_decode=False, token_to_orig_map=None, orig_doc_tokens=None):
        # find beginning of document
        question_end_index = q_len

        # bsz x seqlen => bsz x n_best_size
        start_logits_indices = start_logits.topk(k=self.args.n_best_size, dim=-1).indices
        end_logits_indices = end_logits.topk(k=self.args.n_best_size, dim=-1).indices

        answers = []

        # This loop can't be vectorized, so loop over each example in the batch separetly
        for i in range(start_logits_indices.size(0)):  # bsz
            potential_answers = []
            for start_logit_index in start_logits_indices[i]:  # n_best_size
                for end_logit_index in end_logits_indices[i]:  # n_best_size
                    if start_logit_index <= question_end_index[i]:
                        continue
                    if end_logit_index <= question_end_index[i]:
                        continue
                    if start_logit_index > end_logit_index:
                        continue
                    answer_len = end_logit_index - start_logit_index + 1
                    if answer_len > self.args.max_answer_length:
                        continue
                    potential_answers.append({'start': start_logit_index, 'end': end_logit_index,
                                              'start_logit': start_logits[i][start_logit_index].item(),
                                              'end_logit': end_logits[i][end_logit_index].item()})
            sorted_answers = sorted(potential_answers, key=lambda x: (x['start_logit'] + x['end_logit']), reverse=True)
            if len(sorted_answers) == 0:
                answers.append({'text': 'null', 'score': -1000000})
            else:
                if fancy_span_decode:
                    answer = sorted_answers[0]
                    answer_token_ids = input_ids[i, answer['start']: answer['end'] + 1]
                    orig_doc_start = token_to_orig_map[str(answer['start'].item())]
                    orig_doc_end = token_to_orig_map[str(answer['end'].item())]
                    orig_tokens = orig_doc_tokens[orig_doc_start : (orig_doc_end + 1)]

                    answer_tokens = self._tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())

                    tok_text = self._tokenizer.convert_tokens_to_string(answer_tokens)
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)
                    ## hack, originally noanswer was chosen as answer to questions without answers
                    ## but wordpieces tokenization breaks noanswer to 'no answer' 
                    ## TODO change (this is here only for backward compatibility)
                    if tok_text == ' noanswer':
                        text = 'no'
                    tok_text = tok_text.strip()
                    final_text = get_final_text(tok_text, orig_text, do_lower_case=False, verbose_logging=True)

                    score = answer['start_logit'] + answer['end_logit']
                    answers.append({'text': final_text, 'score': score})
                else:
                    answer = sorted_answers[0]
                    answer_token_ids = input_ids[i, answer['start']: answer['end'] + 1]
                    answer_tokens = self._tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())
                    text = self._tokenizer.convert_tokens_to_string(answer_tokens)
                    ## hack, originally noanswer was chosen as answer to questions without answers
                    ## but wordpieces tokenization breaks noanswer to 'no answer' 
                    ## TODO change
                    if text == ' noanswer':
                        text = 'no'
                    score = answer['start_logit'] + answer['end_logit']
                    answers.append({'text': text, 'score': score})

        if hasattr(self.args, 'question_type_classification_head') and \
                self.args.question_type_classification_head and type_logits is not None:
            type_pred = type_logits.argmax(dim=1)
            assert type_pred.shape[0] == 1
            type_pred = type_pred.tolist()[0]
            if type_pred in [0, 1]:
                answers = [{'text': int_to_answer_type[type_pred], 'score': type_logits.max(dim=1)[0]}]

        # decode supporting facts
        if simple_sentence_decode:
            predicted_sentences = sent_logits.argmax(dim=1)
            supporting_facts = []
            related_sentence_index = []
            sentence_to_score = {}   # {(0, 0): 0.23, (0, 1): -0.43, ...}
            if isinstance(sent_to_par_idx, tuple):
                assert len(sent_to_par_idx) == 1
                sent_to_par_idx = sent_to_par_idx[0]
            for i, v in enumerate(sent_logits[:, 1]):
                sent_par_idx = sent_to_par_idx[str(i)]['par']  # paragraph index of the sentence
                sentence_to_score[(sent_par_idx, i)] = v
            for i, s in enumerate(predicted_sentences):
                if s == 1:
                    supporting_facts.append([sent_to_par_idx[str(i)]['par_title'], sent_to_par_idx[str(i)]['sent']])
                    related_sentence_index.append({'par_index': sent_to_par_idx[str(i)]['par'],
                                                    'sent_index': sent_to_par_idx[str(i)]['sent']})
        else:
            # Decoding to enforce sentences come from exactly 2 paragraph
            par_to_sent_idx = defaultdict(list)  # {0: [0, 1, 2], 1: [3, 4, 5], ...}
            for k, v in sent_to_par_idx.items():
                par_to_sent_idx[v['par']].append(int(k))
            sentence_to_score = {}   # {(0, 0): 0.23, (0, 1): -0.43, ...}
            for i, v in enumerate(sent_logits[:, 1]):
                sent_par_idx = sent_to_par_idx[str(i)]['par']  # paragraph index of the sentence
                # v += 0.2 * par_logits[sent_to_par_idx[0][str(i)]['par']]  # add paragraph score
                sentence_to_score[(sent_par_idx, i)] = v
            par_list = sorted(list(par_to_sent_idx.keys()))
            new_scored_node_sets = {}
            for i, p1 in enumerate(par_list):
                for p2 in par_list[i:]:
                    for ss1 in powerset(self._top_sentences_for_p(p1, sentence_to_score, par_to_sent_idx)):
                        if len(ss1) <= 0:
                            continue
                        ss1_score = sum(sentence_to_score[p1, s1] for s1 in ss1)
                        try:
                            for ss2 in powerset(self._top_sentences_for_p(p2, sentence_to_score, par_to_sent_idx)):
                                if len(ss2) <= 0:
                                    continue
                                ss2_score = sum(sentence_to_score[p2, s2] for s2 in ss2)
                                new_scored_node_sets[frozenset(
                                    {(p1, s1) for s1 in ss1} |
                                    {(p2, s2) for s2 in ss2}
                                )] = ss1_score + ss2_score
                        except KeyError:
                            import ipdb; ipdb.set_trace()
            valid_scored_node_sets = {}
            for k, v in new_scored_node_sets.items():
                pars = {e[0] for e in k}
                if len(pars) != 2:
                    continue
                valid_scored_node_sets[k] = v
            top_sents = sorted(valid_scored_node_sets.items(), key=lambda x: x[1], reverse=True)
            predicted_sentences = [e[1] for e in top_sents[0][0]]
            supporting_facts = []
            related_sentence_index = []
            for s in predicted_sentences:
                supporting_facts.append([sent_to_par_idx[str(s)]['par_title'], sent_to_par_idx[str(s)]['sent']])
                related_sentence_index.append({'par_index': sent_to_par_idx[str(s)]['par'],
                                                'sent_index': sent_to_par_idx[str(s)]['sent']})
        return answers, supporting_facts, related_sentence_index, sentence_to_score

    def sync_list_across_gpus(self, l, device, dtype):
        l_tensor = torch.tensor(l, device=device, dtype=dtype)
        gather_l_tensor = [torch.ones_like(l_tensor) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_l_tensor, l_tensor)
        return torch.cat(gather_l_tensor).tolist()

    def _validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_em = torch.stack([x['val_em'] for x in outputs]).mean()
        string_qids = [item for sublist in outputs for item in sublist['qid']]
        int_qids = [self.val_dataloader_obj.dataset.val_qid_string_to_int_map[qid] for qid in string_qids]
        answer_scores = [item['score'] for sublist in outputs for item in sublist['answers']]
        f1_scores = [item['val_span_f1'] for item in outputs]
        em_scores = [item['val_span_em'] for item in outputs]
        f1_sent_scores = [item['val_sent_f1'] for item in outputs]
        f1_par_scores = [item['val_par_f1'] for item in outputs]
        if outputs[0]['pred_q_type'] is not None:
            f1_q_type = calc_f1(torch.stack([x['pred_q_type'] for x in outputs]), torch.stack([x['q_type_labels'] for x in outputs]))
        else:
            f1_q_type = avg_loss.new_zeros(1).float()
        # print('validation end before sync', len(int_qids), len(answer_scores), len(f1_scores), len(em_scores))
        # print(type(int_qids[0]), type(answer_scores[0]), type(f1_scores[0]), type(em_scores[0]))
        if self.trainer.use_ddp:
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size
            torch.distributed.all_reduce(avg_em, op=torch.distributed.ReduceOp.SUM)
            avg_em /= self.trainer.world_size

            int_qids = self.sync_list_across_gpus(int_qids, avg_loss.device, torch.int)
            answer_scores = self.sync_list_across_gpus(answer_scores, avg_loss.device, torch.float)
            f1_scores = self.sync_list_across_gpus(f1_scores, avg_loss.device, torch.float)
            em_scores = self.sync_list_across_gpus(em_scores, avg_loss.device, torch.int)
            f1_sent_scores = self.sync_list_across_gpus(f1_sent_scores, avg_loss.device, torch.float)
            if self.args.include_paragraph and self.args.paragraph_loss:
                f1_par_scores = self.sync_list_across_gpus(f1_par_scores, avg_loss.device, torch.float)
            torch.distributed.all_reduce(f1_q_type, op=torch.distributed.ReduceOp.SUM)
            f1_q_type /= self.trainer.world_size

        # print('validation end after sync', len(int_qids), len(answer_scores), len(f1_scores), len(em_scores))

        # Because of having multiple documents per questions, some questions might have multiple corresponding answers
        # Here, we only keep the answer with the highest answer_score

        qa_with_duplicates = defaultdict(list)
        for qid, answer_score, f1_score, em_score in zip(int_qids, answer_scores, f1_scores, em_scores):
            qa_with_duplicates[qid].append({'answer_score': answer_score, 'f1': f1_score, 'em': em_score})
        f1_scores = []
        em_scores = []
        for qid, answer_metrics in qa_with_duplicates.items():
            top_answer = sorted(answer_metrics, key=lambda x: x['answer_score'], reverse=True)[0]
            f1_scores.append(top_answer['f1'])
            em_scores.append(top_answer['em'])
        avg_val_f1 = sum(f1_scores) / len(f1_scores)
        avg_val_em = sum(em_scores) / len(em_scores)
        avg_val_sent_f1 = sum(f1_sent_scores) / len(f1_sent_scores)
        if self.args.include_paragraph and self.args.paragraph_loss:
            avg_val_par_f1 = sum(f1_par_scores) / len(f1_par_scores)
        else:
            avg_val_par_f1 = 0.0

        logs = {'val_loss': avg_loss, 'val_f1': avg_val_f1, 'avg_val_f1': avg_val_f1, 'avg_val_sent_f1': avg_val_sent_f1, 'avg_val_em': avg_val_em, 'avg_val_par_f1': avg_val_par_f1, 'avg_f1_q_type': f1_q_type}
        progress_bar = {'avg_val_sent_f1': avg_val_sent_f1, 'avg_val_em': avg_val_em}
        avg_combined_f1 = (avg_val_f1 + avg_val_sent_f1) / 2.0

        return {'avg_val_loss': avg_loss, 'avg_sent_f1': avg_val_sent_f1, 'avg_combined_f1': avg_combined_f1, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_nb):
        return self._validation_step(batch, batch_nb)

    def validation_end(self, outputs):
        return self._validation_end(outputs)

    def _top_sentences_for_p(self, p: int, sentence_to_score: dict, par_to_sents: dict) -> Iterable[int]:
        s_with_score = [(s, sentence_to_score[(p, s)]) for s in par_to_sents[p]]
        s_with_score.sort(key=lambda x: -x[1])
        sub_zero_yielded = 0
        for s, score in s_with_score:
            if score < 0:
                if sub_zero_yielded >= 2:
                    break
                else:
                    sub_zero_yielded += 1
            yield s

    def test_step(self, batch, batch_nb):
        q_id, q_tokens, doc_tokens, start_pos, end_pos, answer_str, sent_labels, num_sents, num_pars, pars, sent_idx, par_offsets, sent_to_par_idx, par_labels, q_type_label, token_to_orig_map, orig_doc_tokens, entity_attention, q_lens, doc_lens, sent_lens, par_lens = batch
        n_batch = len(q_tokens)
        assert n_batch == 1  # doesn't currently support larger batch size
        assert len(sent_to_par_idx) == 1
        output = self.forward(q_tokens, q_lens.max(), doc_tokens, sent_labels, start_pos, end_pos, par_labels, q_type_label, entity_attention=entity_attention)
        sent_logits = output['sent_logits']
        start_logits = output['start_logits']
        end_logits = output['end_logits']
        par_logits = output['par_logits']
        input_ids = torch.cat([q_tokens, doc_tokens], dim=1)
        # answers, supporting_facts, related_sentence_index, sentence_to_score = self.decode(input_ids, start_logits, end_logits, q_lens, sent_logits, par_logits, pars, sent_to_par_idx, simple_sentence_decode=False)
        ret = {
            'q_id': q_id[0],
            'input_ids': input_ids,
            'start_logits': start_logits,
            'end_logits': end_logits,
            'q_lens': q_lens,
            'sent_logits': sent_logits, 
            'par_logits': par_logits,
        }
        return ret

    def test_end(self, outputs):
        if self.args.test_file_orig:
            with open(self.args.test_file_orig) as fin:
                test_gold = json.load(fin)
                sp_gold_by_id = {e['_id']: e['supporting_facts'] for e in test_gold}
        sp = {}
        answer = {}
        related_sent_index = {} # a dict of key, values where values are a list of {'par_index': 1, 'sent_index': 0} tuples
        sentence_scores = {}
        par_scores = {}
        int_qids = [self.test_dataloader_obj.dataset.val_qid_string_to_int_map[e['q_id']] for e in outputs if e is not None]
        input_ids = [e['input_ids'] for e in outputs if e is not None]
        start_logits = [e['start_logits'] for e in outputs if e is not None]
        end_logits = [e['end_logits'] for e in outputs if e is not None]
        q_lens = [e['q_lens'] for e in outputs if e is not None]
        sent_logits = [e['sent_logits'] for e in outputs if e is not None]
        par_logits = [e['par_logits'] for e in outputs if e is not None]
        if outputs and outputs[0].get('type_logits'):
            type_logits = [e['type_logits'] for e in outputs if e is not None]
        else:
            type_logits = [None for _ in outputs]
        if len(int_qids) != len(outputs):
            print(f'Warning: None outputs in test step: {len(int_qids)} of {len(outputs)}')
        par_by_id = {}

        for _int_qid, _input_id, _start_logit, _end_logit, _q_len, _sent_logit, _par_logit, _type_logit in \
                tqdm(zip(int_qids, input_ids, start_logits, end_logits, q_lens, sent_logits, par_logits, type_logits),
                     total=len(int_qids),
                     disable=(self.trainer.use_ddp and self.trainer.proc_rank > 0), desc='decoding...'):

            qid = self.test_dataloader_obj.dataset[_int_qid][0]
            assert qid not in answer
            pars = self.test_dataloader_obj.dataset[_int_qid][9]
            sent_to_par_idx = self.test_dataloader_obj.dataset[_int_qid][12]
            token_to_orig_map = self.test_dataloader_obj.dataset[_int_qid][15]
            orig_doc_tokens = self.test_dataloader_obj.dataset[_int_qid][16]
            answers, supporting_facts, related_sentence_index, sentence_to_score = self.decode(
                _input_id, _start_logit, _end_logit, _q_len, _sent_logit, _par_logit, _type_logit, pars, sent_to_par_idx, 
                simple_sentence_decode=args.simple_sentence_decode, fancy_span_decode=self.args.fancy_decode, orig_doc_tokens=orig_doc_tokens,
                token_to_orig_map=token_to_orig_map
            )
            par_score = _par_logit[:, 1].tolist()
            par_scores[qid] = par_score
            par_by_id[qid] = pars
            answer[qid] = answers[0]['text']
            sp[qid] = supporting_facts
            related_sent_index[qid] = related_sentence_index
            sentence_scores[qid] = {f'{ee[0]}-{ee[1]}': vv.item() for ee, vv in sentence_to_score.items()}

        predictions = {'answer': answer, 'sp': sp, 'sentence_scores': sentence_scores, 'par_scores': par_scores, 'par_by_id': par_by_id}

        pathlib.Path(self.args.test_output_dir).mkdir(parents=True, exist_ok=True)

        if self.trainer.use_ddp:
            # create intermediate files for each worker output
            predictions_file = self.args.test_output_dir + f'/prediction-output_rank-{self.trainer.proc_rank}.json'
            with open(self.args.test_output_dir + f'/related-sentence-index_rank-{self.trainer.proc_rank}.json', 'w') as f_out:
                json.dump(related_sent_index, f_out)
            with open(predictions_file, 'w') as f_out:
                json.dump(predictions, f_out)

            torch.distributed.barrier()

            if self.trainer.proc_rank == 0:
                all_predictions = {'answer': {}, 'sp': {}, 'sentence_scores': {}, 'par_scores': {}, 'par_by_id': {}}
                all_related_sent_index = {}
                for rank in range(self.trainer.world_size):
                    with open(self.args.test_output_dir + f'/prediction-output_rank-{rank}.json') as fin:
                        preds = json.load(fin)
                        for k in all_predictions:
                            all_predictions[k].update(preds[k])
                    with open(self.args.test_output_dir + f'/related-sentence-index_rank-{rank}.json') as fin:
                        preds = json.load(fin)
                        all_related_sent_index.update(preds)
                predictions_file = self.args.test_output_dir + f'/prediction-output.json'

                metrics = None
                if self.args.test_file_orig:
                    metrics = hotpot_evaluate(all_predictions, test_gold, hotpot_format=True, allow_partial_prediction=False)
                    metrics_file = self.args.test_output_dir + f'/metrics.json'
                    with open(metrics_file, 'w') as f_out:
                        json.dump(metrics, f_out)
                        print(json.dumps(metrics, indent=2))
                with open(self.args.test_output_dir + '/related-sentence-index.json', 'w') as f_out:
                    json.dump(all_related_sent_index, f_out)
                with open(predictions_file, 'w') as f_out:
                    json.dump(all_predictions, f_out)
                return metrics
            else:
                return {}
        else:
            predictions_file = self.args.test_output_dir + f'/prediction-output.json'

            metrics = None
            if self.args.test_file_orig:
                metrics = hotpot_evaluate(predictions, test_gold, hotpot_format=True, allow_partial_prediction=False)
                metrics_file = self.args.test_output_dir + f'/metrics.json'
                with open(metrics_file, 'w') as f_out:
                    json.dump(metrics, f_out)
                print(json.dumps(metrics, indent=2))
                print('\t'.join(['em', 'f1', 'sp_em', 'sp_f1', 'joint_em', 'joint_f1']))
                print('\t'.join([str(metrics['em']), str(metrics['f1']), str(metrics['sp_em']), str(metrics['sp_f1']), str(metrics['joint_em']), str(metrics['joint_f1'])]))
            with open(self.args.test_output_dir + '/related-sentence-index.json', 'w') as f_out:
                json.dump(related_sent_index, f_out)
            with open(predictions_file, 'w') as f_out:
                json.dump(predictions, f_out)

            return metrics

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # batch_nb is batch number in the current epoch
        # it is incremented for each batch, not for each gradient update
        optimizer.step()
        optimizer.zero_grad()
        self._num_grad_updates += 1
        if self.args.optimizer_type == 'fairseq_optimizer':
            self.scheduler.step_update(self._num_grad_updates)
        else:
            self.scheduler.step(self._num_grad_updates)


    def configure_optimizers(self):
        if self.args.optimizer_type == 'fairseq_optimizer':
            from fairseq import optim
            from fairseq.optim.lr_scheduler import build_lr_scheduler

            if hasattr(self.args, 'fp16') and self.args.fp16:
                self.half()

            # TODO: handle restart
            self.roberta.args.lr = [self.args.lr]
            if hasattr(self.args, 'fp16') and self.args.fp16:
                self.half()
                optimizer = optim.FP16Optimizer.build_optimizer(self.roberta.args, list(self.parameters()))
                print("Using fp16")
                print(optimizer)
            else:
                optimizer = optim.build_optimizer(self.roberta.args, self.parameters())

            self.roberta.args.lr_scheduler = 'polynomial_decay'
            self.roberta.args.reset_lr_scheduler = True
            self.roberta.args.warmup_updates = self.args.warmup
            self.roberta.args.end_learning_rate = 0.0
            self.roberta.args.min_lr = 0.0
            gpu_count = max(self.args.total_gpus, 1)
            grad_accum_steps = self.args.batch_size if self.args.model_type == 'longformer' else self.args.grad_accum  # This model doesn't work with batch size greater than 1
            if self.args.model_type == 'tvm_roberta' or self.args.model_type == 'longformer':
                self.args.total_num_updates = (self.args.num_epochs * self.args.dataset_size / self.args.batch_size / gpu_count)
            else:
                self.args.total_num_updates = (self.args.num_epochs * self.args.dataset_size / self.args.grad_accum / gpu_count)
            print(f'\n\n-------\n effective batch size: {gpu_count * 1 * grad_accum_steps}\n'
                f'total num updates: {self.args.total_num_updates}\n\n----\n')
            self.roberta.args.power = 1.0
            self.scheduler = build_lr_scheduler(self.roberta.args, optimizer)

            if self.args.resume_from_checkpoint is not None:
                checkpoint = torch.load(self.args.resume_from_checkpoint, map_location='cpu')
                self._num_grad_updates = int(checkpoint['global_step'] / checkpoint['hparams']['batch_size'])
            else:
                self._num_grad_updates = 0
            if self.args.ignore_scheduler_state:
                self._num_grad_updates = 0
            else:
                self.scheduler.step_update(self._num_grad_updates)

            return [optimizer]
        elif self.args.optimizer_type == 'adam':
            grad_accum_steps = self.args.batch_size  # This model doesn't work with batch size greater than 1
            gpu_count = max(self.args.total_gpus, 1)
            self.args.total_num_updates = (self.args.num_epochs * self.args.dataset_size / self.args.batch_size / gpu_count)
            print(f'\n\n-------\n effective batch size: {gpu_count * 1 * grad_accum_steps}\n'
                f'total num updates: {self.args.total_num_updates}\n\n----\n')

            def lr_lambda(current_step):
                if current_step < self.args.warmup:
                    return float(current_step) / float(max(1, self.args.warmup))
                return max(0.0, float(self.args.total_num_updates - current_step) / float(max(1, self.args.total_num_updates - self.args.warmup)))
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)  # scheduler is not saved in the checkpoint, but global_step is, which is enough to restart
            self.scheduler.step(self.global_step / self.hparams.batch_size)

            if self.args.resume_from_checkpoint is not None:
                checkpoint = torch.load(self.args.resume_from_checkpoint, map_location='cpu')
                self._num_grad_updates = int(checkpoint['global_step'] / checkpoint['hparams']['batch_size'])
            else:
                self._num_grad_updates = 0

            return optimizer
        else:
            assert False

    def _get_loader(self, split):
        fname = os.path.join(self.args.train_file if split=='train' else self.args.dev_file)
        is_train = split == 'train'

        dataset = HotpotDataset(fname, max_seq_len=self.args.max_seq_len, num_samples=self.args.num_samples, split=split)

        if self.args.total_gpus > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            batch_size = self.args.batch_size if self.args.model_type == 'roberta' else 1
            loader = DataLoader(
                dataset, batch_size=batch_size, num_workers=self.args.num_workers,
                sampler=sampler, collate_fn=custom_collate)
        else:
            # batch size should be 1, we accumulate gradients
            batch_size = self.args.batch_size if self.args.model_type == 'roberta' else 1
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=is_train, num_workers=self.args.num_workers, collate_fn=custom_collate)
        return loader

    def train_dataloader(self):
        return self._get_loader('train')

    def val_dataloader(self):
        if self.val_dataloader_obj is not None:
            return self.val_dataloader_obj
        self.val_dataloader_obj = self._get_loader('dev')
        return self.val_dataloader_obj

    def test_dataloader(self):
        if self.test_dataloader_obj is not None:
            return self.test_dataloader_obj
        self.test_dataloader_obj = self._get_loader('dev')
        return self.test_dataloader_obj  # hotpotqa doesn't have a separate test set


    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--save-dir", type=str, required=True)
        parser.add_argument("--save-prefix", type=str, required=True)
        parser.add_argument("--train-file", type=str, required=True)
        parser.add_argument("--dev-file", type=str, required=True)

        parser.add_argument("--model-type", type=str, required=True)
        parser.add_argument("--model-path", type=str, required=True)
        parser.add_argument("--model-filename", type=str, required=False, help='if loading roberta')

        parser.add_argument("--num-gpus", type=str, default=1)
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
        parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
        parser.add_argument("--val-check-interval", type=int, default=250, help="number of gradient updates between checking validation loss")
        parser.add_argument("--version", type=int, default=0, help="Run id for checkpointing")

        parser.add_argument("--warmup", type=int, default=200, help="Number of warmup steps")
        parser.add_argument("--val-percent-check", default=0.2, type=float)
        parser.add_argument("--test-percent-check", default=1.0, type=float)
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--max-seq-len", type=int, default=4032)
        parser.add_argument("--seed", type=int, default=1234, help="Seed")

        parser.add_argument('--force-n2-attention', default=False, action='store_true')
        parser.add_argument("--local-rank", "--local_rank", default=-1, type=int)

        parser.add_argument('--extra-attn', default=False, action='store_true')
        parser.add_argument('--create-new-weight-matrics', default=False, action='store_true')
        # parser.add_argument('--symmetric-candidate-attention', default=False, action='store_true')  # always true
        parser.add_argument('--separate-full-attention-projection', default=False, action='store_true')

        parser.add_argument('--use-segment-ids', default=False, action='store_true')

        parser.add_argument("--num-samples", type=int, default=None)
        parser.add_argument("--test-only", default=False, action='store_true')
        parser.add_argument("--test-checkpoint", default=None, help='path to the model output')
        parser.add_argument("--test-output-dir", default=None, help='path to the output directory for writing predictions')
        parser.add_argument("--test-file-orig", default=None, help='if running in test mode provide this (original format of hotpot)')
        parser.add_argument("--fp16", default=False, action='store_true')
        parser.add_argument("--or-softmax-loss", default=False, action='store_true')
        parser.add_argument("--mixing-ratio", default=0.2, type=float, help='mixing ratio for multitask loss')
        parser.add_argument("--n-best-size", default=20, type=int, help='number of answer candidates, for decoding')
        parser.add_argument("--max_answer_length", type=int, default=20, help="maximum num of wordpieces/answer. Used at decoding time")
        parser.add_argument("--grad-accum", default=1, type=int)
        parser.add_argument("--dynamic-mixing-ratio", default=False, action='store_true')
        parser.add_argument("--include-paragraph", default=False, action='store_true', help='include paragraph in extra attention')
        parser.add_argument("--paragraph-loss", default=False, action='store_true', help='include paragraph in extra attention')
        parser.add_argument("--multi-layer-classification-heads", default=False, action='store_true', help='include paragraph in extra attention')
        parser.add_argument("--mlp-qa-head", default=False, dest='mlp_qa_head', action='store_true', help='include paragraph in extra attention')
        parser.add_argument("--mixing-ratio-par", default=0.2, type=float, help='mixing ratio for paragraph loss')
        parser.add_argument("--resume-from-checkpoint", default=None, help='To resume training from a specific checkpoint pass in the path here')
        parser.add_argument("--ignore-scheduler-state", default=False, action='store_true')
        parser.add_argument("--ignore-optimizer-state", default=False, action='store_true')
        parser.add_argument("--initialize-from-checkpoint", default=None, help='Init model state with checkpoint')
        parser.add_argument("--dataset-size", default=90447, type=int)
        parser.add_argument("--question-type-classification-head", default=False, action='store_true', help='add a head for classifying question types, yes/no/span')
        parser.add_argument("--loss-at-different-layers", default=False, action='store_true', help='Attach sentence/span loss at different layers (hardcoded to 20 and 24)')
        parser.add_argument("--linear-mixing", default=None, help='mixing ratios for span/type/sent/par respectively e.g., [0.25,1,1,0.5]')
        parser.add_argument("--fancy-decode", default=False, action='store_true')
        parser.add_argument("--simple-sentence-decode", default=False, action='store_true')
        parser.add_argument("--overrides", default=None, help='override model args when loading a checkpoint, a json string')
        parser.add_argument("--attention-mode", choices={'n2', 'tvm', 'sliding_chunks'}, default='sliding_chunks')

        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--optimizer-type", choices={'fairseq_optimizer', 'adam'}, default='adam')

        parser.add_argument("--include-entities", default=False, action='store_true')

        return parser



def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if ',' in args.num_gpus:
        args.num_gpus = list(map(int, args.num_gpus.split(',')))
        args.total_gpus = len(args.num_gpus)
    else:
        args.num_gpus = int(args.num_gpus)
        args.total_gpus = args.num_gpus

    if args.test_only:
        print('loading model...')
        overrides = {'model_path': args.model_path, 'num_gpus': args.num_gpus, 'total_gpus': args.total_gpus}
        print(overrides)
        if args.overrides:
            for k, v in json.loads(args.overrides).items():
                overrides[k] = v
        model = HotpotModel.load_from_checkpoint(
            args.test_checkpoint,
            overrides=overrides
        )
        model.args = args
        model.args.num_gpus = args.num_gpus  # TODO: add support for ddp in test
        model.args.total_gpus = args.total_gpus
        model.args.attention_window = 256  # ugly hack, when loading the model from pretrained the attention window is not loaded, need to manually set
        model.args.attention_mode = 'sliding_chunks'
        model.args.dev_file = args.dev_file
        model.args.test_file = args.dev_file
        model.args.train_file = args.dev_file  # the model won't get trained, pass in the dev file instead to load faster
        trainer = Trainer(gpus=args.num_gpus, test_percent_check=args.test_percent_check, train_percent_check=0.01, val_percent_check=0.01,
                          distributed_backend='ddp' if args.total_gpus > 1 else None)
        trainer.test(model)

    else:
        model = HotpotModel(args)

        # logger here is a SummaryWritter for tensorboard
        # it is used by the trainer, and certain return variables
        # from the model are automatically logged
        logger = TestTubeLogger(
            save_dir=args.save_dir,
            name=args.save_prefix,
            version=args.version
        )

        if args.mixing_ratio > 0 and not args.linear_mixing:
            monitor_metric = 'avg_combined_f1'
        elif args.mixing_ratio == 0 and not args.linear_mixing:
            monitor_metric = 'avg_sent_f1'
        elif args.mixing_ratio == 1 or (args.linear_mixing and json.loads(args.linear_mixing)[2] == 0):  # sentence loss is zero
            monitor_metric = 'avg_val_em'
        else:
            monitor_metric = 'avg_combined_f1'


        ckpt_filepath = f"{args.save_dir}/{logger.name}/version_{logger.version}/checkpoints"
        checkpoint_callback = ModelCheckpoint(
            # model saved to filepath/prefix_....
            filepath= ckpt_filepath + '/{epoch:02d}-{avg_combined_f1:.3f}-{avg_sent_f1:.3f}-{avg_val_em:.3f}',
            prefix='',
            save_top_k=3,
            verbose=True,
            monitor=monitor_metric,
            mode='max',
            period=-1
        )

        if args.initialize_from_checkpoint:
            print(f'initalizing from checkpoint: {args.initialize_from_checkpoint}')
            checkpoint = torch.load(args.initialize_from_checkpoint, map_location=lambda storage, loc: storage)
            is_dp_module = isinstance(model, (LightningDistributedDataParallel,
                                                LightningDataParallel))
            if is_dp_module:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            print('weights initalized')

        trainer = Trainer(gpus=args.num_gpus, distributed_backend='ddp' if args.total_gpus > 1 else None,
                            track_grad_norm=-1,
                            accumulate_grad_batches=args.batch_size if args.model_type in ['tvm_roberta', 'longformer'] else args.grad_accum,
                            max_epochs=args.num_epochs, early_stop_callback=None,
                            val_check_interval=args.val_check_interval * args.batch_size if args.model_type=='tvm_roberta' else args.val_check_interval,
                            logger=logger,
                            val_percent_check=args.val_percent_check,
                            checkpoint_callback=checkpoint_callback,
                            resume_from_checkpoint=args.resume_from_checkpoint,
                            use_amp=not args.fp32, amp_level='O2')

        trainer.fit(model)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser = HotpotModel.add_model_specific_args(parser)
    args = parser.parse_args()
#    args = args_class()
    main(args)
