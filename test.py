# !/usr/bin/env python3
from typing import List
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

from model import UIE, convert_inputs, get_bool_ids_greater_than, get_span

device = 'cuda:2'
model = torch.load('./checkpoints/model_best/model.pt')
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/model_best")


def inference(contents: List[str], prompts: List[str], max_length=512, prob_threshold=0.4) -> List[str]:
    """
    输入 promot 和 content 列表，返回模型提取结果。    

    Args:
        contents (List[str]): 待提取文本列表, e.g. -> [
                                                    '《琅琊榜》是胡歌主演的一部电视剧。',
                                                    '《笑傲江湖》是一部金庸的著名小说。',
                                                    ...
                                                ]
        prompts (List[str]): prompt列表，用于告知模型提取内容, e.g. -> [
                                                                    '主语',
                                                                    '类型',
                                                                    ...
                                                                ]
        max_length (int): 句子最大长度，小于最大长度则padding，大于最大长度则截断。
        prob_threshold (float): sigmoid概率阈值，大于该阈值则二值化为True。

    Returns:
        List: 模型识别结果, e.g. -> [['琅琊榜'], ['电视剧']]
    """
    inputs = convert_inputs(tokenizer, prompts, contents, max_length=50)
    model_inputs = {
        'input_ids': inputs['input_ids'].to(device),
        'token_type_ids': inputs['token_type_ids'].to(device),
        'attention_mask': inputs['attention_mask'].to(device),
    }
    output_sp, output_ep = model(**model_inputs)
    output_sp, output_ep = output_sp.detach().cpu().tolist(), output_ep.detach().cpu().tolist()
    start_ids_list = get_bool_ids_greater_than(output_sp, prob_threshold)
    end_ids_list = get_bool_ids_greater_than(output_ep, prob_threshold)

    res = []                                                    # decode模型输出，将token id转换为span text
    offset_mapping = inputs['offset_mapping'].tolist()
    for start_ids, end_ids, prompt, content, offset_map in zip(start_ids_list, 
                                                            end_ids_list,
                                                            prompts,
                                                            contents,
                                                            offset_mapping):
        span_set = get_span(start_ids, end_ids)                 # e.g. {(5, 7), (9, 10)}
        current_span_list = []
        for span in span_set:
            if span[0] < len(prompt) + 2:                       # 若答案出现在promot区域，过滤
                continue
            span_text = ''                                      # 答案span
            input_content = prompt + content                    # 对齐token_ids
            for s in range(span[0], span[1] + 1):               # 将 offset map 里 token 对应的文本切回来
                span_text += input_content[offset_map[s][0]: offset_map[s][1]]
            current_span_list.append(span_text)
        res.append(current_span_list)
    return res


def event_extract_example():
    """
    事件抽取示例。
    """
    sentence = "6月10日加班打车回家25元"
    schema = {'出行触发词': ['时间','花费','目的地']}
    
    rsp = {}
    trigger_prompt = list(schema.keys())[0]

    trigger = inference([sentence], [trigger_prompt])[0]
    rsp[trigger_prompt] = trigger
    if trigger:
        arguments = schema.get(trigger_prompt)
        contents = [sentence] * len(arguments)
        prompts = ["{}的{}".format(trigger, a) for a in arguments]
        res = inference(contents, prompts)
        for a, r in zip(arguments, res):
            rsp[a] = r
    print(rsp)


def ner_example():
    """
    NER任务示例。
    """
    schema = ['主语', '类型']
    sentences = [
                    '杨小卫，汉族，陕西宝鸡市陈仓区人，生于1990年3月。大专文化，中共预备党员。', 
                    '杨小卫，汉族，陕西宝鸡市陈仓区人，生于1990年3月。大专文化，中共预备党员。'
            ]
    res = inference(sentences, schema, max_length=128)
    print(res)


if __name__ == "__main__":
    # ner_example()
    event_extract_example()

