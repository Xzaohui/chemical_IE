import json
import numpy as np
import model_bertcrf
import torch
import datetime
from transformers import BertTokenizer,BertTokenizerFast
import bert_data_manage
from torch.utils.data import Dataset, DataLoader

bert_path='./model/chemical-bert-uncased-negative'
tokenizer_word=BertTokenizer.from_pretrained(bert_path)
tokenizer = BertTokenizerFast.from_pretrained(bert_path)
model=model_bertcrf.model_bertcrf()
model.load_state_dict(torch.load("./model/bertcrf_negative.pkl"))

model.to("cuda")
model.eval()


def predict(data_input):
    t=tokenizer([data_input],truncation=True,padding=True,return_offsets_mapping=True,max_length=512,return_tensors="pt")
    total_data=t['input_ids']
    total_mask=t['attention_mask']
    offset_mapping=t['offset_mapping'].numpy()

    total_data=total_data.cuda().long()
    total_mask=total_mask.cuda().long()

    total_offset_mapping=torch.tensor(offset_mapping).cuda().long()
    class predict_dataset(Dataset):
        def __init__(self,total_data,total_mask,total_offset_mapping):
            self.total_data=total_data
            self.total_mask=total_mask
            self.total_offset_mapping=total_offset_mapping
        def __len__(self):
            return len(self.total_data)
        def __getitem__(self,idx):
            return self.total_data[idx],self.total_mask[idx],self.total_offset_mapping[idx]

    pre_dataset=predict_dataset(total_data,total_mask,total_offset_mapping)
    pre_dataloader=DataLoader(pre_dataset,batch_size=1,shuffle=True ,num_workers = 0)
    f= open('./res.txt','r+')
    for i,(test_data,attention_mask,total_offset_mapping) in enumerate(pre_dataloader):
        _,path=model(test_data,attention_mask)
        j=1
        res="{\"label\":["
        while(j<len(path)):
            if path[j]==0:
                m=tokenizer_word.convert_ids_to_tokens(test_data[0][j].tolist())
                start=total_offset_mapping[0][j][0].item()
                end=total_offset_mapping[0][j][1].item()
                j+=1
                while(j<len(path) and path[j]==1):
                    tmp=tokenizer_word.convert_ids_to_tokens(test_data[0][j].tolist())
                    if tmp.startswith("##"):
                        m+=tmp[2:]
                    else:
                        m+=" "+tmp
                    end=total_offset_mapping[0][j][1].item()
                    j+=1
                res+="["+str(start)+","+str(end)+",\""+m+"\",\"M\"],"
                continue
            if path[j]==2:
                m=tokenizer_word.convert_ids_to_tokens(test_data[0][j].tolist())
                start=total_offset_mapping[0][j][0].item()
                end=total_offset_mapping[0][j][1].item()
                j+=1
                while(j<len(path) and path[j]==3):
                    tmp=tokenizer_word.convert_ids_to_tokens(test_data[0][j].tolist())
                    if tmp.startswith("##"):
                        m+=tmp[2:]
                    else:
                        m+=" "+tmp
                    end=total_offset_mapping[0][j][1].item()
                    j+=1
                res+="["+str(start)+","+str(end)+",\""+m+"\",\"R\"],"
                continue
            if path[j]==4:
                m=tokenizer_word.convert_ids_to_tokens(test_data[0][j].tolist())
                start=total_offset_mapping[0][j][0].item()
                end=total_offset_mapping[0][j][1].item()
                j+=1
                while(j<len(path) and path[j]==5):
                    tmp=tokenizer_word.convert_ids_to_tokens(test_data[0][j].tolist())
                    if tmp.startswith("##"):
                        m+=tmp[2:]
                    else:
                        m+=" "+tmp
                    end=total_offset_mapping[0][j][1].item()
                    j+=1
                res+="["+str(start)+","+str(end)+",\""+m+"\",\"P\"],"
                continue
            if path[j]==6:
                m=tokenizer_word.convert_ids_to_tokens(test_data[0][j].tolist())
                start=total_offset_mapping[0][j][0].item()
                end=total_offset_mapping[0][j][1].item()
                j+=1
                while(j<len(path) and path[j]==7):
                    tmp=tokenizer_word.convert_ids_to_tokens(test_data[0][j].tolist())
                    if tmp.startswith("##"):
                        m+=tmp[2:]
                    else:
                        m+=" "+tmp
                    end=total_offset_mapping[0][j][1].item()
                    j+=1
                res+="["+str(start)+","+str(end)+",\""+m+"\",\"F\"],"
                continue
            j+=1
        res=res.strip(',')+"]}"
        f.read()
        f.write(res+'\n')
        print(res)
    f.close()


def test_average_score():
    with torch.no_grad():
        score=torch.zeros(1).to('cuda')
        i=0
        for i,(test_data,test_lab,attention_mask) in enumerate(bert_data_manage.test_dataloader):
            path_score,path_index=model(test_data,attention_mask)
            score+=path_score
        print(score/i)

def train_average_score():
    with torch.no_grad():
        score=torch.zeros(1).to('cuda')
        i=0
        for i,(test_data,test_lab,attention_mask) in enumerate(bert_data_manage.train_dataloader):
            path_score,path_index=model(test_data,attention_mask)
            score+=path_score
        print(score/i)

def p_r_f():
    with torch.no_grad():
        cont_total=np.zeros((2,len(bert_data_manage.BIO_lab)))
        cont_cur=np.zeros((1,len(bert_data_manage.BIO_lab)))
        for i,(test_data,test_lab,attention_mask) in enumerate(bert_data_manage.test_dataloader):
                path_score,path_index=model(test_data,attention_mask)
                test_lab=test_lab[0].cpu().numpy()
                for j in range(len(path_index)):
                    cont_total[0][test_lab[j]]+=1
                    cont_total[1][path_index[j]]+=1
                    if test_lab[j]==path_index[j]:
                        cont_cur[0][path_index[j]]+=1
        i=0

        for i in range(len(bert_data_manage.BIO_lab)-2):
            precision=cont_cur[0][i]/cont_total[1][i]
            recall=cont_cur[0][i]/cont_total[0][i]
            print("label:",list(bert_data_manage.BIO_lab.keys())[list(bert_data_manage.BIO_lab.values()).index(i)],"精确率:",precision)
            print("label:",list(bert_data_manage.BIO_lab.keys())[list(bert_data_manage.BIO_lab.values()).index(i)],"召回率:",recall)
            print("label:",list(bert_data_manage.BIO_lab.keys())[list(bert_data_manage.BIO_lab.values()).index(i)],"F1:",2*precision*recall/(precision+recall))
            print("-----------------------------------------")
        print("总的精确率:",cont_cur[0].sum()/cont_total[1].sum())
        print("总的召回率:",cont_cur[0].sum()/cont_total[0].sum())
        print("总的F1:",2*cont_cur[0].sum()/(cont_total[1].sum()+cont_total[0].sum()))
        print("=================================================")

def imp_p_r_f(model):
    cont_total=np.zeros((2,len(bert_data_manage.BIO_lab)))
    cont_cur=np.zeros((1,len(bert_data_manage.BIO_lab)))
    for i,(test_data,test_lab,attention_mask) in enumerate(bert_data_manage.test_dataloader):
        path_score,path_index=model(test_data,attention_mask)
        test_lab=test_lab[0].cpu().numpy()
        for j in range(len(path_index)):
            cont_total[0][test_lab[j]]+=1
            cont_total[1][path_index[j]]+=1
            if test_lab[j]==path_index[j]:
                cont_cur[0][path_index[j]]+=1
    
    return 2*cont_cur[0,0:8].sum()/(cont_total[1,0:8].sum()+cont_total[0,0:8].sum())

def average_p_r_f(model,dataloader):
    cont_total=np.zeros((2,len(bert_data_manage.BIO_lab)))
    cont_cur=np.zeros((1,len(bert_data_manage.BIO_lab)))
    for i,(test_data,test_lab,attention_mask) in enumerate(dataloader):
        path_score,path_index=model(test_data,attention_mask)
        test_lab=test_lab[0].cpu().numpy()
        for j in range(len(path_index)):
            cont_total[0][test_lab[j]]+=1
            cont_total[1][path_index[j]]+=1
            if test_lab[j]==path_index[j]:
                cont_cur[0][path_index[j]]+=1
    
    return 2*cont_cur[0,0:8].sum()/(cont_total[1,0:8].sum()+cont_total[0,0:8].sum())


# p_r_f()
# train_average_score()
# test_average_score()
# print(imp_p_r_f(model))
# predict()