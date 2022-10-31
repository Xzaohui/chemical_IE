import numpy as np
import model_bertcrf
import torch
import datetime
from transformers import BertTokenizer,get_linear_schedule_with_warmup
import bert_data_manage

tokenizer_word=BertTokenizer.from_pretrained(bert_data_manage.bert_path)
model=model_bertcrf.model_bertcrf()



def train(model,dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5,weight_decay=0.01)
    total_steps = len(dataloader) * 1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=total_steps)
    model.to("cuda")
    model.train()
    for j in range(2):
        for i,(train_data,train_lab,attention_mask) in enumerate(dataloader):
            optimizer.zero_grad()
            loss=model.loss(train_data,attention_mask,train_lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if i%10==0:
                print(datetime.datetime.now())
                print(loss)
                print("epoch:",j,"step:",i)
                print("=================================================")
train(model,bert_data_manage.train_dataloader)
torch.save(model.state_dict(), "/home/chemical_IE/model/bertcrf_"+datetime.datetime.now().strftime('%m-%d,%H_%M_%S')+".pkl")
model.save_bert(bert_data_manage.bert_path)
print("=================saved model===========================")
print(datetime.datetime.now())

