import json
f=open('5-noB_our_last.jsonl','r')
prompt_data=open('prompt_data.jsonl','a')
prompt_match={'M':'materials type','R':'regulatory factor','P':'product category','F':'faradaic eﬀiciency','CO2':'carbon dioxide','CA':'catalyst','ELE':'electroreduction','RE':'reduction'}
# tdata={}
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
                tdata['result_list']+=[{'text':line['data'][label[0]:label[1]],'start':label[0],'end':label[1]}]
                del_list.append(labels.index(label))
        prompt_data.write(json.dumps(tdata,ensure_ascii=False)+'\n')
        for i in reversed(del_list):
            del labels[i]
    break