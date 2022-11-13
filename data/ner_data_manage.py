import json
import transformers
from transformers import AutoTokenizer,BertTokenizerFast
prompt_match={'M':'materials type','R':'regulatory factor','P':'product category','F':'faradaic eﬀiciency','CO2':'carbon dioxide','CA':'catalyst','ELE':'electroreduction','RE':'reduction'}
tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
def data2prompt_manage(nerdata_path,prompt_data_path):
    f=open(nerdata_path,'r')
    prompt_data=open(prompt_data_path,'a')
    for line in f.readlines():
        line = json.loads(line)
        labels=line['label']
        while labels:
            del_list=[]
            tdata={}
            tdata['content']=line['data']
            tdata['result_list']=[]
            tdata['prompt']=prompt_match[labels[0][2]]
            for label in labels:
                if label[2]==labels[0][2]:
                    tdata['result_list']+=[{'text':label[3],'start':label[0],'end':label[1]}] # 同一个prompt下的所有实体，prompt词是本身的意思
                    del_list.append(labels.index(label))
            prompt_data.write(json.dumps(tdata,ensure_ascii=False)+'\n')
            for i in reversed(del_list):
                del labels[i]
        # break

def data_pre_manage(origin_path,managed_path):
    f=open(origin_path,'r')
    for line in f.readlines():
        line = json.loads(line)
        chemical_data=line['data']
        ner_data=line['label']
        offest_mapping(chemical_data,ner_data,managed_path)
        # break


def offest_mapping(chemical_str,ner_mapping,managed_path):
    tokens=tokenizer(text=chemical_str,return_offsets_mapping=True)
    if len(tokens['input_ids'])>512:
        mid=len(tokens['input_ids'])//2
        mid=chemical_str[:mid].rfind('.')
        ner_mapping_ids=len(ner_mapping)//2
        while ner_mapping[ner_mapping_ids][1]>mid:
            ner_mapping_ids-=1
        while ner_mapping[ner_mapping_ids][1]<mid:
            ner_mapping_ids+=1
        offest_mapping(chemical_str[:mid],ner_mapping[:ner_mapping_ids],managed_path)
        offest_mapping(chemical_str[mid:],ner_mapping[ner_mapping_ids:],managed_path)
        print(1)
        return
    offest_mapping=[]
    managed_data_dump=open(managed_path,'a')
    ner_ids=0
    ner_begin_ids=0
    ner_end_ids=0
    for i,map in enumerate(tokens['offset_mapping'][1:-1]):
        # print(map)
        # print(ner_mapping[ner_ids])
        if ner_ids>=len(ner_mapping):
            break
        if map[0]==ner_mapping[ner_ids][0]:
            ner_begin_ids=i
        if map[1]==ner_mapping[ner_ids][1]:
            ner_end_ids=i+1
            offest_mapping.append([ner_begin_ids,ner_end_ids,ner_mapping[ner_ids][2],chemical_str[ner_mapping[ner_ids][0]:ner_mapping[ner_ids][1]]])
            ner_ids+=1
    tdata={}
    tdata['data']=chemical_str
    tdata['label']=offest_mapping
    managed_data_dump.write(json.dumps(tdata)+'\n')
    return offest_mapping


if __name__ == '__main__':
    # data_pre_manage('5-noB_our_last.jsonl','data_managed.jsonl')
    data2prompt_manage('data_managed.jsonl','prompt_data.jsonl')
    # print(tokenizer(text='materials type',return_offsets_mapping=True))
    # print(tokenizer(text='materials type',truncation=True,padding=True,return_offsets_mapping=True,max_length=512,return_tensors="pt"))