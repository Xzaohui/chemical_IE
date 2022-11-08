# !/usr/bin/env python3
"""
torch 版本 UIE fintuning 脚本。

Author: pankeyu
Date: 2022/09/06
"""
import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler

from metrics import SpanEvaluator
from model import UIE, convert_example
from iSummaryWriter import iSummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, type=str, required=True, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoint", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=300, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--log_name", default='log.png', type=str, help="log image name.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


writer = iSummaryWriter(log_path='./logs', log_name=args.log_name)


def evaluate(model, metric, data_loader, global_step):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    model.eval()
    metric.reset()

    for batch in data_loader:
        start_prob, end_prob = model(input_ids=batch['input_ids'].to(args.device),
                                        token_type_ids=batch['token_type_ids'].to(args.device),
                                        attention_mask=batch['attention_mask'].to(args.device))
        start_ids = batch['start_ids'].to(torch.float32).detach().numpy()
        end_ids = batch['end_ids'].to(torch.float32).detach().numpy()
        num_correct, num_infer, num_label = metric.compute(start_prob.cpu().detach().numpy(), 
                                                            end_prob.cpu().detach().numpy(), 
                                                            start_ids, 
                                                            end_ids)
        metric.update(num_correct, num_infer, num_label)
    
    precision, recall, f1 = metric.accumulate()
    writer.add_scalar('eval-precision', precision, global_step)
    writer.add_scalar('eval-recall', recall, global_step)
    writer.add_scalar('eval-f1', f1, global_step)
    writer.record()
    
    model.train()
    return precision, recall, f1


def train():
    model = AutoModel.from_pretrained(args.model)
    model = UIE(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    print(dataset['train'])
    convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)
    model.to(args.device)

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    tic_train = time.time()
    metric = SpanEvaluator()
    criterion = torch.nn.BCELoss()
    global_step, best_f1 = 0, 0

    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            # print('batch: ', batch)
            start_prob, end_prob = model(input_ids=batch['input_ids'].to(args.device),
                                        token_type_ids=batch['token_type_ids'].to(args.device),
                                        attention_mask=batch['attention_mask'].to(args.device))
            start_ids = batch['start_ids'].to(torch.float32).to(args.device)
            end_ids = batch['end_ids'].to(torch.float32).to(args.device)
            loss_start = criterion(start_prob, start_ids)
            loss_end = criterion(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                tokenizer.save_pretrained(cur_save_dir)

                precision, recall, f1 = evaluate(model, metric, eval_dataloader, global_step)
                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    best_f1 = f1
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                    tokenizer.save_pretrained(cur_save_dir)
                tic_train = time.time()


if __name__ == '__main__':
    train()
