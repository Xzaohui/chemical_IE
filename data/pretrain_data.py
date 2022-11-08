import json

def pretrain_data_manage(data_path, save_path):
    metadata=open(data_path,'r')
    pre_train=open(save_path,'a')
    b=0
    i=1
    for data in metadata.readlines():
        b=0
        data=json.loads(data)
        tmp=data.get('title','')
        if tmp:
            b=1
            pre_train.write(tmp.replace('[','').replace(']','')+'\n')
        tmp=data.get('abstract','')
        if tmp:
            b=1
            if type(tmp)==list:
                for section in tmp:
                    for sentence in section['text'].split('. '):
                        if sentence not in ['.','']:
                            pre_train.write(sentence.replace('[','').replace(']','')+'.\n')
            else:
                for sentence in tmp.split('. '):
                    if sentence not in ['.','']:
                        pre_train.write(sentence.replace('[','').replace(']','')+'.\n')
        tmp=data.get('body_text','')
        if tmp:
            b=1
            if type(tmp)==list:
                for section in tmp:
                    for sentence in section['text'].split('. '):
                        if sentence not in ['.','']:
                            pre_train.write(sentence.replace('[','').replace(']','')+'.\n')
            else:
                for sentence in tmp.split('. '):
                    if sentence not in ['.','']:
                        pre_train.write(sentence.replace('[','').replace(']','')+'.\n')
        if b==1:
            pre_train.write('\n')
        i+=1
        if i%100==0:
            break

if __name__=="__main__":
    # for i in range(1):
    #     pretrain_data_manage('metadata_{}.jsonl'.format(str(i)),'pretrain_data.txt')
    for i in range(1):
        pretrain_data_manage('pdf_parses_{}.jsonl'.format(str(i)),'pretrain_data.txt')