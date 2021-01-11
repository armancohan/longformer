""" preprocess hotpotqa dataset
Data format in hotpot qa is:

{'supporting_facts': List[Tuple[pargraph title, sentence_id]],
 'level': str,
 'question': str,
 'context': List[Tuple[paragraph title, List[sentences]]],
 'answer': str,
 '_id': str,
 'type': str}

"""

import argparse
import itertools
import json
import operator
import pathlib
import random
import time
import functools
from collections import defaultdict
from functools import lru_cache
from multiprocessing.pool import Pool
import re
import spacy
from spacy.tokens import Doc
import string

from tqdm.auto import tqdm

_spacy_nlp = None

# tokens from gpt2 vocab
# Q_START = 'madeupword0000'
# Q_END = 'madeupword0001'

Q_START = '[question]'
Q_END = '[/question]'
TITLE_START = '<t>'
TITLE_END = '</t>'
# SENT_MARKER = '[sent]'
# SENT_MARKER_END = '[/sent]'
SENT_MARKER = '<s>'
SENT_MARKER_END = '</s>'
PAR = '[/par]'
DOC_START = '<doc-s>'
DOC_END = '</doc-s>'

def normalize_string(s):
    s = s.replace(' .', '.')
    s = s.replace(' ,', ',')
    s = s.replace(' !', '!')
    s = s.replace(' ?', '?')
    s = s.replace('( ', '(')
    s = s.replace(' )', ')')
    s = s.replace(" 's", "'s")
    return ' '.join(s.strip().split())


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


class WhitespaceTokenizer(object):
    # custom tokenizer for spacy pipeline
    def __init__(self, vocab):
        self.vocab = vocab
    def __call__(self, text):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        # special tokenization to find answer spans in the doc
        # first whitespace tokenize and then tokenize each word using bpe
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
        spaces = [True] * len(doc_tokens)
        return Doc(self.vocab, words=doc_tokens, spaces=spaces)


def find_string_in_wordpieces(word_pieces, s, tokenizer, lowercase=False):
    # word_pieces = [list of word piece strings]
    # s = a string
    # lowercase = should we lowercase for comparison?
    #
    # returns [[start_index1, end_index1], [start_index2, end_index2], ..]
    #   where s == word_pieces[start_index:(end_index+1)] detokenized
    if lowercase:
        s = s.lower()
    s = normalize_string(s)

    def get_s_piece(sp):
        s_piece = tokenizer.convert_tokens_to_string(sp)
        if lowercase:
            s_piece = s_piece.lower()
        s_piece_utf = s_piece.encode('utf-8')
        if s_piece_utf == b'\xef\xbf\xbd' or s_piece_utf == b' \xef\xbf\xbd':
            return None
        else:
            return s_piece

    matches = []
    piece_no = 0
    while piece_no < len(word_pieces):
        piece = word_pieces[piece_no]
        s_piece = get_s_piece(piece)
        start = piece_no
        if s_piece is None:
            s_piece = get_s_piece(word_pieces[piece_no:(piece_no+2)])
            piece_no += 1
        if s_piece is None:
            s_piece = get_s_piece(word_pieces[piece_no:(piece_no+3)])
        for k in range(len(s_piece)):
            if s_piece[k:] == s[:len(s_piece[k:])]:
                candidate_match = s_piece[k:]
                end = piece_no
                while len(candidate_match) < len(s) and end < len(word_pieces) - 1:
                    end += 1
                    next_piece = get_s_piece(word_pieces[end])
                    if next_piece is None:
                        next_piece = get_s_piece(word_pieces[end:(end+2)])
                        if next_piece is None:
                            next_piece = get_s_piece(word_pieces[end:(end+3)])
                        end += 1
                    candidate_match += next_piece
                    if candidate_match != s[:len(candidate_match)]:
                        break
                if candidate_match[:len(s)] == s:
                    matches.append([start, end])
                    break
        piece_no += 1

    return matches

@lru_cache(maxsize=1)
def get_roberta_tokenizer(seq_len=999999999):
    from transformers.tokenization_roberta import RobertaTokenizer

    additional_tokens = [Q_START, Q_END, TITLE_START, TITLE_END, SENT_MARKER_END, SENT_MARKER, PAR, DOC_START, DOC_END]
    # for i in range(max_candidates):
    #     additional_tokens.extend(['[ent{}]'.format(i), '[/ent{}]'.format(i)])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    tokenizer.add_tokens(additional_tokens)
    # tokenizer.max_len = seq_len
    # tokenizer.max_len_single_sentence = seq_len
    # tokenizer.max_len_sentences_pair = seq_len

    return tokenizer


def get_supporting_sentences(supporting_facts, context):
    """
    Gets supporting sentences from context
    Args:
        supporting facts: List[List[paragraph title, sentence index]]
        context: List[paragraph title, List[sentences]]
    Returns:
        list of labels, list of actual sentences
    """
    # first create a flat list of all sentences corresponding to their paragraph
    context_flat = []
    for e in context:
        for i, ee in enumerate(e[1]):
            context_flat.append((ee, e[0], i))
    # context_flat = [(ee, e[0], i) for e in context for i, ee in enumerate(e[1])]
    # create a map from paragraph title, sent index to the index of context)
    context_map = {f'{e[1]}-{e[2]}': i for i, e in enumerate(context_flat)}
    context_map_reverse = {v: k for k, v in context_map.items()}
    labels = [0 for _ in range(len(context_flat))]
    support_sents = []
    for par, sent_id in supporting_facts:
        key = f'{par}-{sent_id}'
        try:
            labels[context_map[key]] = 1
            support_sents.append(context_flat[context_map[key]][0])
        except KeyError:
            print('Key error in label mapping, skipping')
    return {'support_labels': labels,
            'support_sents': support_sents,
            'context_flat': context_flat,
            'paragraph_to_sentence_id': context_map,
            'sentence_id_to_paragraph': context_map_reverse}

def strip_unicode(s):
    ids = [ord(c) for c in s]
    return ''.join([chr(i) for i in ids if i < 256])

def process_instance_for_support_prediction_baseline(instance):
    tokenizer = get_roberta_tokenizer()
    def tok(s):
        return tokenizer.tokenize(normalize_string(s), add_prefix_space=True)
    question_tokens = [Q_START] + tok(instance['question']) + [Q_END]
    sentences = []
    supporting_facts = {e[0]: e[1] for e in instance['supporting_facts']}
    # paragraphs = {i: ' '.join(e[1]) for i, e in enumerate(instance['context'])}

    positives = []
    negatives = []
    for par in instance['context']:
        if par[0] not in supporting_facts:
            negatives.append(par)
        else:
            positives.append(par)
    random.shuffle(negatives)
    negatives = negatives[:2]

    candidate_paragraphs = negatives + positives
    random.shuffle(candidate_paragraphs)


    for par_idx, par in enumerate(candidate_paragraphs):
        par_title = par[0]
        for sent_idx, sent in enumerate(par[1]):
            try:
                if sent_idx == supporting_facts[par_title]:
                    is_support = True
                else:
                    is_support = False
            except KeyError:
                is_support = False
            sentences.append([par_idx, par_title, sent_idx, tok(sent), is_support])
    sentences_with_full_context = []
    for par_idx, par in enumerate(candidate_paragraphs):
        par_title = par[0]
        paragraph_sents = {}
        for idx in range(len(par[1])):
            current_sent = [SENT_MARKER] + tok(par[1][idx]) + [SENT_MARKER_END]
            try:
                if idx == supporting_facts[par_title]:
                    is_support = True
                else:
                    is_support = False
            except KeyError:
                is_support = False
            paragraph_sents[idx] = current_sent
            for idx2, s2 in enumerate(par[1]):
                if idx == idx2:
                    continue
                paragraph_sents[idx2] = tok(s2)
            par_text = [e[1] for e in sorted(paragraph_sents.items(), key=operator.itemgetter(0))]
            question_id = f"{instance['_id']}-{par_idx}-{idx}"
            sentences_with_full_context.append([par_idx, par_title, par_text, is_support, question_id])
    return {'question': question_tokens,
            'answer': tok(instance.get('answer')),
            'sentences': sentences,
            'sentences_with_full_context': sentences_with_full_context,
            'question_id': question_id}


def _shorten_context(paragraphs, current_len, max_tokens, num_sents, max_sents, predict_mode=False):
    """ shorten a paragraph by removing negative sentences """
    tries = 0
    while current_len > max_tokens and tries < 100:
        i = random.randrange(len(paragraphs))
        if 'is_relevant' in paragraphs[i] and not predict_mode and paragraphs[i]['is_relevant']:
            continue
        sent = paragraphs[i]['sentences'][-1]
        if len(paragraphs[i]['sentences']) == 1:
            continue
        del paragraphs[i]['sentences'][-1]
        del paragraphs[i]['sentence_labels'][-1]
        current_len -= len(sent)
        num_sents -= 1
        tries += 1
    while num_sents > max_sents and tries < 100:
        i = random.randrange(len(paragraphs))
        if 'is_relevant' in paragraphs[i] and not predict_mode and paragraphs[i]['is_relevant']:
            continue
        sent = paragraphs[i]['sentences'][-1]
        if len(paragraphs[i]['sentences']) == 1:
            continue
        del paragraphs[i]['sentences'][-1]
        del paragraphs[i]['sentence_labels'][-1]
        current_len -= len(sent)
        num_sents -= 1
        tries += 1
    if tries == 100:
        print('something wrong, shortening context failed with 100 tries of removing sentences')
    return paragraphs, current_len, num_sents


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)


def process_instance_for_support_prediction(instance, max_sents, max_tokens, shorten_long_context, train, include_answer,
                                            ignore_par_title, new_version, add_doc_separators, add_bos_token,
                                            process_ner):
    """
    Creates a long context of form
    <sep> q q q </sep> s1 s1 s1 </sent> s2 s2 s2 </sent> ....
    """
    global _spacy_nlp

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    tokenizer = get_roberta_tokenizer()
    def tok(s):
        return tokenizer.tokenize(normalize_string(s), add_prefix_space=True)

    # question string
    question_str = tokenizer.cls_token + ' ' + Q_START + ' ' + instance['question'] + ' ' + Q_END + ' '

    if instance.get('answer') and include_answer:
        question_str += instance['answer'] + ' ' + tokenizer.mask_token  # + ' ' + tokenizer.sep_token  [not include sep]
    else:
        question_str += tokenizer.mask_token # + ' ' + tokenizer.sep_token   [not include sep]

    question_tokens = tok(question_str)
    q_len = len(question_tokens)

    # process supporting facts
    sentences = []
    if instance.get('supporting_facts'):
        supporting_facts = defaultdict(set)
        for e in instance['supporting_facts']:
            supporting_facts[e[0]].add(e[1])
    else:
        supporting_facts = None
    # paragraphs = {i: ' '.join(e[1]) for i, e in enumerate(instance['context'])}

    relevant_par_idx = []
    sent_labels = []
    sent_cnt = 0
    par_cnt = 0
    paragraph_sents = []
    total_context_len = 0

    # find supporting evidences and put them into a list of dicts
    # hotpotqa paragraphs are in the following format:
    # [['paragraph title', ['sentence 1', 'sentence 2' ]], ...]
    # gold supporting facts are a list of paragraph titles with sentence index
    sent_to_par_index = {}
    for i, par in enumerate(instance['context']):
        par_dict = {}
        if supporting_facts is not None:
            par_dict['is_relevant'] = 1 if par[0] in supporting_facts else 0

        title_str = TITLE_START + ' ' + par[0] + ' ' + TITLE_END + ' '
        par_dict['title'] = title_str
        par_dict['sentences'] = []
        par_dict['sentence_labels'] = []
        for j, sent in enumerate(par[1]):
            sent_to_par_index[sent_cnt] = {'par': i, 'sent': j, 'par_title': par[0]}
            if add_bos_token:
                sent_str = SENT_MARKER + ' ' + sent + ' ' + SENT_MARKER_END
            else:
                sent_str = sent + ' ' + SENT_MARKER_END
            par_dict['sentences'].append(sent_str)
            if supporting_facts is not None:
                try:
                    if j in supporting_facts[par[0]]:
                        par_dict['sentence_labels'].append(1)
                    else:
                        par_dict['sentence_labels'].append(0)
                except KeyError:
                    par_dict['sentence_labels'].append(0)
            sent_cnt += 1
        paragraph_sents.append(par_dict)

    # shorten long context
    if shorten_long_context and total_context_len > max_tokens or sent_cnt > max_sents:
        paragraph_sents, total_context_len, num_sents = _shorten_context(
            paragraph_sents, total_context_len, max_tokens - 3, sent_cnt, max_sents, predict_mode=False)
    else:
        num_sents = sent_cnt
    sentence_labels = []

    sent_cnt = 0
    document_str = ''
    # create the document string
    for p in paragraph_sents:
        # add title
        if add_doc_separators:
            document_str += DOC_START
        document_str += ' ' + p['title']
        # add sentences
        if p.get('sentence_labels'):
            for sent, label in zip(p['sentences'], p['sentence_labels']):
                sent_cnt += 1
                document_str += sent
                sentence_labels.append(label)
        else:
            for sent in p['sentences']:
                sent_cnt += 1
                document_str += sent
        if add_doc_separators:
            document_str += DOC_END

    if supporting_facts is not None:
        relevant_par_idx = [i for i, e in enumerate(paragraph_sents) if e['is_relevant'] == 1]
        par_labels = [e['is_relevant'] for e in paragraph_sents]
    else:
        par_labels = []

    # include additional `null yes no` tokens
    # additional_str = ' ' + PAR + ' null yes no' + tokenizer.sep_token
    additional_str = ''
    document_str += additional_str

    if sentence_labels:
        assert sent_cnt == len(sentence_labels) == num_sents

    # starting and ending character index for sentences
    sent_indices_end_char = [m.end() for m in re.finditer(SENT_MARKER_END.replace('[', '\[').replace(']', '\]'), document_str)]
    sent_indices_start_char = [len(question_str)] + [sent_indices_end_char[i] for i in range(len(sent_indices_end_char) - 1)]
    # starting and ending character index for sentences
    par_indices_start = [m.start() for m in re.finditer(TITLE_START, document_str)]
    par_indices_end = [e - 1 for e in par_indices_start[1:]] + [len(document_str) - len(additional_str)]

    par_title_end_indices = [m.end() for m in re.finditer(TITLE_END, document_str)]
    assert len(par_indices_start) == len(par_title_end_indices)
    par_title_boundaries = list(zip(par_indices_start, par_title_end_indices))

    # find answer string in the document string
    if instance.get('answer'):
        answer_start_candidates = [(m.start(), m.end()) for m in re.finditer(re.escape(instance['answer']), document_str)]
        if instance['answer'] in ['no', 'yes']:
            answer_start_candidates = answer_start_candidates[-1:]

        # valid answer spans by sentence (span is valid if it appears in gold sentence)
        valid_answer_spans = [(sent_indices_start_char[i], sent_indices_end_char[i]) for i, e in enumerate(sentence_labels) if e == 1] + [(len(document_str)-len(additional_str), len(document_str))]
        # valid answer spans by paragraph (span is valid if it appears in gold evidence paragraph)
        valid_answer_spans_par = [(par_indices_start[i], par_indices_end[i]) for i in relevant_par_idx] + [(len(document_str)-len(additional_str), len(document_str))]

        # final valid spans
        final_spans = []
        for start, end in answer_start_candidates:
            for span_start, span_end in valid_answer_spans:
                if start >= span_start and end <= span_end:
                    if ignore_par_title:  # if answer span is within paragraph title boundaries continue
                        matched_to_par_title = False
                        for e1, e2 in par_title_boundaries:
                            if start >= e1 and end < e2:
                                matched_to_par_title = True
                                break
                        if matched_to_par_title:
                            continue
                    final_spans.append((start, end))
        if len(final_spans) == 0:
            for start, end in answer_start_candidates:
                for span_start, span_end in valid_answer_spans_par:
                    if start >= span_start and end <= span_end:
                        if ignore_par_title:  # if answer span is within paragraph title boundaries continue
                            matched_to_par_title = False
                            for e1, e2 in par_title_boundaries:
                                if start >= e1 and end < e2:
                                    matched_to_par_title = True
                                    break
                            if matched_to_par_title:
                                continue
                        final_spans.append((start, end))
        if len(final_spans) == 0 and ignore_par_title:  # this time if the answer is par title it is fine
            for start, end in answer_start_candidates:
                for span_start, span_end in valid_answer_spans_par:
                    if start >= span_start and end <= span_end:
                        final_spans.append((start, end))

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    # special tokenization to find answer spans in the doc
    # first whitespace tokenize and then tokenize each word using bpe
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in document_str:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    if _spacy_nlp is None and process_ner:
        print('loading spacy')
        # spacy.prefer_gpu()
        _spacy_nlp = spacy.load("en_core_web_md")
        # _spacy_nlp.tokenizer = WhitespaceTokenizer(_spacy_nlp.vocab)
    entity_set = set()
    if process_ner:
        entity_locations = []
        entity_locations_char = []
        doc = _spacy_nlp(document_str)
        # assert len(doc) == len(doc_tokens)
        for ent in doc.ents:
            for loc in range(ent.start, ent.end):
                if ent.text not in [Q_START, Q_END, TITLE_START, TITLE_END, SENT_MARKER_END, SENT_MARKER, PAR]:
                    entity_locations.append(loc)
                entity_set.add(ent.text)
            entity_locations_char.append([ent.start_char, ent.end_char])
    else:
        entity_locations = []
        entity_locations_char = []

    if instance.get('answer') and process_ner:
        if instance.get('answer') in entity_set:
            covered_by_entity = 1
        else:
            covered_by_entity = 0
    else:
        covered_by_entity = 0


    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    entity_attention = []
    entity_locations_set = set(entity_locations)
    sub_token_counter = 0
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        # ugly hack: the line below should have been `self.tokenizer.tokenize(token')`
        # but roberta tokenizer uses a different subword if the token is the beginning of the string
        # or in the middle. So for all tokens other than the first, simulate that it is not the first
        # token by prepending a period before tokenizing, then dropping the period afterwards
        sub_tokens = tokenizer.tokenize(f'. {token}')[1:] if i > 0 else tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
            if i in entity_locations_set:
                entity_attention.append(1)
            else:
                entity_attention.append(0)
            sub_token_counter += 1
    assert len(entity_attention)  == sub_token_counter == len(all_doc_tokens)


    if instance.get('answer') and train:
        final_spans = [_improve_answer_span(all_doc_tokens, e[0], e[1], tokenizer, instance['answer']) for e in final_spans]

    # starting and ending sentence indices in the tokenized document
    sent_indices_end = [i for i, e in enumerate(all_doc_tokens) if e == SENT_MARKER_END]
    sent_indices_start = [0] + [e + 1 for e in sent_indices_end[:-1]]
    # starting and ending paragraph indices in the tokenized document
    par_indices_start = [i for i, e in enumerate(all_doc_tokens) if e == TITLE_START]
    additional_str_start = len(all_doc_tokens)
    for i in range(len(all_doc_tokens)- 1, 0, -1):
        if all_doc_tokens[i] == PAR:
            additional_str_start = i
            break
    par_indices_end = [e - 1 for e in par_indices_start[1:]] + [additional_str_start]

    if instance.get('answer'):
        answer_spans = []
        start_positions, end_positions = [], []
        if instance.get('answer') in ['yes', 'no']:
            start_positions, end_positions = [0], [0]
        else:
            for answer_char_offset_start, answer_char_offset_end in final_spans:
                start_position = char_to_word_offset[answer_char_offset_start]
                if new_version:  # in newer version this was modified (also in decoding time)
                    end_position = char_to_word_offset[min(answer_char_offset_end - 1, len(char_to_word_offset) - 1)]
                else:
                    end_position = char_to_word_offset[min(answer_char_offset_end, len(char_to_word_offset) - 1)]
                answer_spans.append((start_position, end_position))
                tok_start_position_in_doc = orig_to_tok_index[start_position]
                not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
                tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc
                start_positions.append(tok_start_position_in_doc)
                end_positions.append(tok_end_position_in_doc)

        assert len(start_positions) == len(end_positions)
    else:
        start_positions = end_positions = []

    if process_ner:
        ner_spans = []
        is_entity = [False for _ in range(len(all_doc_tokens))]
        for ner_char_offset_start, ner_char_offset_end in entity_locations_char:
            start_pos = char_to_word_offset[ner_char_offset_start]
            end_position = char_to_word_offset[min(ner_char_offset_end - 1, len(char_to_word_offset) - 1)]
            tok_start_position_in_doc = orig_to_tok_index[start_pos]
            not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
            tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc
            ner_spans.append((tok_start_position_in_doc, tok_end_position_in_doc))
            for offset_ in range(tok_start_position_in_doc, tok_end_position_in_doc + 1):
                is_entity[offset_] = True
    else:
        is_entity = []

    # if len(all_doc_tokens) > max_tokens:
    #     print(f'document long even after shortening {len(all_doc_tokens)}, discarding the example')
    #     return None
    if not start_positions and train:
        print('no answer found, skipping')
        return None, None

    token_to_orig_map = {}
    for i in range(len(all_doc_tokens)):
        index = len(question_tokens) + i
        token_to_orig_map[index] = tok_to_orig_index[i]

    assert len(paragraph_sents) == len(instance["context"])
    obj = {'q_id': instance['_id'],
           'q_tokens': question_tokens,
           'doc_tokens': all_doc_tokens,
           'start_pos': start_positions,
           'end_pos': end_positions,
           'answer_str': instance.get('answer'),
           'sent_labels': sentence_labels,
           'par_labels': par_labels,
           'num_sents': sent_cnt,
           'num_pars': len(paragraph_sents),
           'pars': paragraph_sents,
           'sent_idx': list(zip(sent_indices_start, sent_indices_end)),
           'par_idx': list(zip(par_indices_start, par_indices_end)),
           'sent_to_par_idx': sent_to_par_index,
           'token_to_orig_map': token_to_orig_map,
           'orig_doc_tokens': doc_tokens,
           'entity_attention': is_entity
    }
    return obj, covered_by_entity


def _find_answer_in_context(context, answer, tokenizer):
    """ find sublist in list """
    idx = []
    for i in range(len(context)):
        if context[i] == answer[0] and tokenizer.convert_tokens_to_string(answer) in tokenizer.convert_tokens_to_string(context[i:i+len(answer)]):
            idx.append(i)
    return idx


def process_instance(instance, max_sents, max_tokens, shorten_long_context, train, include_answer,
                     ignore_par_title, new_version, add_doc_separators, add_bos_token, process_entities):
    processed, covered = process_instance_for_support_prediction(
        instance, max_sents, max_tokens, shorten_long_context=shorten_long_context,
        train=train, include_answer=include_answer, ignore_par_title=ignore_par_title, new_version=new_version,
        add_doc_separators=add_doc_separators, add_bos_token=add_bos_token, process_ner=process_entities)
    return {'instance': processed, 'covered': covered}

def tok(s):
    return get_roberta_tokenizer(100000000).tokenize(normalize_string(s), add_prefix_space=True)

def preprocess_hotpot(args):

    with open(args.input, 'r') as fin:
        data = json.load(fin)
    print("Read data, {} instances".format(len(data)))

    if args.answer_prediction_file is not None:
        # replace answers with predicted answers
        with open(args.answer_prediction_file, 'r') as fin:
            pred_answers = json.load(fin)
            if 'sp' in pred_answers.keys() and 'answer' in pred_answers.keys():
                pred_answers = pred_answers['answer']
            cnt_missing = 0
            for i, d in enumerate(data):
                try:
                    data[i]['answer'] = pred_answers[d['_id']]
                except KeyError:
                    data[i]['answer'] = 'null'
                    cnt_missing += 1
            print(f'{cnt_missing} answers are empty')

    if args.sentence_mode is not None:
        if args.sentence_mode == 'baseline':
            process_fn = process_instance_for_support_prediction_baseline
        elif args.sentence_mode == 'tvm':
            process_fn = functools.partial(process_instance_for_support_prediction, max_sents=args.max_sents, max_tokens=args.max_tokens, include_answer=args.include_answer, ignore_par_title=args.ignore_par_title)
    else:
        process_fn = functools.partial(process_instance, max_sents=args.max_sents, max_tokens=args.max_tokens, shorten_long_context=args.shorten_long_context, train=args.train, include_answer=args.include_answer, ignore_par_title=args.ignore_par_title, new_version=args.new_version, add_doc_separators=args.add_doc_separators, add_bos_token=args.add_bos_token, process_entities=args.process_entities)
    if args.workers > 1:
        with Pool(args.workers) as p:
            processed_results = list(tqdm(p.imap(process_fn, data), total=len(data), unit_scale=1))
    else:
        processed_results = [process_fn(d) for d in tqdm(data)]

    new_data = [e['instance'] for e in processed_results if e['instance'] is not None]
    print(f'skipped: {len(processed_results) - len(new_data)} of {len(processed_results)}')
    covered_by_entity = [e['covered'] for e in processed_results if e['covered'] is not None]
    print(f"covered: {sum(covered_by_entity)}/{len(covered_by_entity)}: {sum(covered_by_entity)/len(covered_by_entity):.4f}")

    with open(args.output, 'w') as fout:
        for d in tqdm(new_data, desc='writing to file'):
            if d:
                fout.write(f'{json.dumps(d)}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='')
    parser.add_argument('output')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--sentence-mode', default=None, help='`baseline`, `tvm`')
    # arguments below are only relevant for sentence_mode=='tvm'
    parser.add_argument('--max-sents', default=1000, help='maximum number of sentences in the context', type=int)
    parser.add_argument('--include-answer', default=False, action='store_true', help='include answer in the context')
    parser.add_argument('--max-tokens', default=4092, help='max number of tokens in the context', type=int)
    parser.add_argument('--shorten-long-context', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--answer-prediction-file', default=None, help='`path`, default is None, if included the gold answer will be replace by the predicted answer spans')
    parser.add_argument('--ignore-par-title', default=False, action='store_true', help='do not include answer positions that are paragraph titles')
    parser.add_argument('--no-bos-token', action='store_true', help='do not add <s> token')
    parser.add_argument('--no-doc-sep', action='store_true', help='do not add document separators')
    parser.add_argument('--no-entities', action='store_true', help='do not extract named entities')
    parser.add_argument('--old-version', default=False, action='store_true', help='a later version of preprocessing that has 1 end offset difference and the new decoding expects this.'

                                                                                   'use this with newer model (model without spans in par titles)')
    random.seed(2)
    args = parser.parse_args()

    args.add_bos_token = False if args.no_bos_token else True
    args.add_doc_separators = False if args.no_doc_sep else True
    args.new_version = False if args.old_version else True
    args.process_entities = False if args.no_entities else True

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preprocess_hotpot(args)

if __name__ == '__main__':
    main()
