# !/usr/bin/env python3
import json
from typing import List

import torch
import torch.nn as nn
import numpy as np


class UIE(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden_size = 768
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor,
        attention_mask=None,
        pos_ids=None,
    ) -> tuple:
        sequence_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]
        start_logits = self.linear_start(sequence_output)       # (batch, seq_len, 1)
        start_logits = torch.squeeze(start_logits, -1)          # (batch, seq_len)
        start_prob = self.sigmoid(start_logits)                 # (batch, seq_len)
        end_logits = self.linear_end(sequence_output)           # (batch, seq_len, 1)
        end_logits = torch.squeeze(end_logits, -1)              # (batch, seq_len)
        end_prob = self.sigmoid(end_logits)                     # (batch, seq_len)
        return start_prob, end_prob


def get_bool_ids_greater_than(probs: list, limit=0.5, return_prob=False) -> list:
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_span(start_ids: list, end_ids: list, with_prob=False) -> set:
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            if start_ids[start_pointer][0] == end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer][0] < end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer][0] > end_ids[end_pointer][0]:
                end_pointer += 1
                continue
        else:
            if start_ids[start_pointer] == end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer] < end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer] > end_ids[end_pointer]:
                end_pointer += 1
                continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def convert_inputs(tokenizer, prompts: List[str], contents: List[str], max_length=512) -> dict:
    inputs = tokenizer(text=prompts,                 # [SEP]?????????
                        text_pair=contents,          # [SEP]?????????
                        truncation=True,             # ????????????
                        max_length=max_length,       # ??????????????????
                        padding="max_length",        # padding??????
                        return_offsets_mapping=True, # ??????offsets????????????token_id??????????????????
                )
    pos_ids = []
    for i in range(len(contents)):
        pos_ids += [[j for j in range(len(inputs['input_ids'][i]))]]
    pos_ids = torch.tensor(pos_ids)
    inputs['pos_ids']=pos_ids

    offset_mappings = [[list(x) for x in offset] for offset in inputs["offset_mapping"]]
    for i in range(len(offset_mappings)):                           # offset ?????????
        bias = 0
        for index in range(1, len(offset_mappings[i])):
            mapping = offset_mappings[i][index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = offset_mappings[i][index - 1][1]
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            offset_mappings[i][index][0] += bias
            offset_mappings[i][index][1] += bias
    
    inputs['offset_mapping'] = offset_mappings

    for k, v in inputs.items():                                     # list???tensor
        inputs[k] = torch.LongTensor(v)

    return inputs


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def convert_example(examples, tokenizer, max_seq_len):  # ???????????????????????????
    """
    ??????????????????????????????????????????????????????

    Args:
        examples (dict): ??????????????????, e.g. -> {
                                                "text": [
                                                            {
                                                                "content": "????????????????????????", 
                                                                "prompt": "??????",
                                                                "result_list": [{"text": "??????", "start": 0, "end": 2}]
                                                            },
                                                        ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]], 
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'pos_ids': [[0, 1, 2, ...], [0, 1, 2, ...], ...],
                            'start_ids': [[0, 1, 0, ...], [0, 0, ..., 1, ...]],
                            'end_ids': [[0, 0, 1, ...], [0, 0, ...]]
                        }
    """
    tokenized_output = {
            'input_ids': [], 
            'token_type_ids': [],
            'attention_mask': [],
            'pos_ids': [],
            'start_ids': [],
            'end_ids': []
        }

    for example in examples['text']:
        example = json.loads(example)
        try:
            encoded_inputs = tokenizer(
                text=example['prompt'],
                text_pair=example['content'],
                stride=len(example['prompt']),
                truncation=True,
                max_length=max_seq_len,
                padding='max_length',
                return_offsets_mapping=True)
        except:
            print('[Warning] ERROR Sample: ', example)
            exit()
        offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
        bias = 0
        for index in range(len(offset_mapping)):
            if index == 0:
                continue
            mapping = offset_mapping[index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = index
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            offset_mapping[index][0] += bias
            offset_mapping[index][1] += bias

        start_ids = [0 for x in range(max_seq_len)]
        end_ids = [0 for x in range(max_seq_len)]

        for item in example["result_list"]:
            start = map_offset(item["start"] + bias, offset_mapping)    # ???????????????start token???id
            end = map_offset(item["end"] - 1 + bias, offset_mapping)    # ???????????????end token???id
            start_ids[start] = 1.0                                      # one-hot vector
            end_ids[end] = 1.0                                          # one-hot vector

        pos_ids = [i for i in range(len(encoded_inputs['input_ids']))]
        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['pos_ids'].append(pos_ids)
        tokenized_output['start_ids'].append(start_ids)
        tokenized_output['end_ids'].append(end_ids)

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v, dtype='int64')

    return tokenized_output