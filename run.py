# -*- coding: utf-8 -*-
import torch
import copy
import torch.nn as nn
import json

from config import DataConfig, ModelConfig
from data_processor import DataProcessor
from bidaf import BidafModel
from loss import LogCriterion
from evaluate import main

def test(model, data, loss_func, config):
    model.eval()
    answers = dict()
    total_loss = 0.
    total_data = 0
    for batch in data.dev_iter:
        c_char = batch.c_char
        c_word = batch.c_word[0]
        c_lens = batch.c_word[1]
        context = (c_word, c_char, c_lens)

        q_char = batch.q_char
        q_word = batch.q_word[0]
        q_lens = batch.q_word[1]
        query = (q_word, q_char, q_lens)

        p1, p2 = model(context, query)

        loss = loss_func((p1, p2), (batch.s_idx, batch.e_idx))

        total_loss += loss.item()

        score, s_idx = p1.max(dim=-1)
        score, e_idx = p2.max(dim=-1)

        batch_size = p1.size(0)
        total_data += batch_size
        for i in range(batch_size):
            id = batch.id[i]
            answer = c_word[i][s_idx[i]:e_idx[i]+1]
            answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
            answers[id] = answer
            
    json_str = json.dumps(answers, indent=4, ensure_ascii=False)
    with open(config.prediction_file, 'w') as f:
        f.write(json_str)

    results = main(config)
    loss = total_loss / (total_data + 1e-10)
    return loss, results['exact_match'], results['f1']

def train(model, data, optimizer, loss_func, config):
    train_iter = data.train_iter
    total_loss = 0.
    total_data = 0
    max_dev_f1 = 0.
    best_model_params = copy.deepcopy(model.state_dict())
    model.train()
    for i, batch in enumerate(train_iter):
        c_char = batch.c_char
        c_word = batch.c_word[0]
        c_lens = batch.c_word[1]
        context = (c_word, c_char, c_lens)

        q_char = batch.q_char
        q_word = batch.q_word[0]
        q_lens = batch.q_word[1]
        query = (q_word, q_char, q_lens)

        s_idx, e_idx = batch.s_idx, batch.e_idx

        present_epoch = int(train_iter.epoch)
        if present_epoch == config.epochs:
            break

        p1, p2 = model(context, query)
        total_data += c_char.size(0)
        loss = loss_func((p1, p2), (s_idx, e_idx))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % config.print_freq == 0:
            train_loss = total_loss / (total_data + 1e-10)
        
            dev_loss, dev_exact, dev_f1 = test(model, data, loss_func, config)
            print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f}'
                  f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')
            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1
                best_model_params = copy.deepcopy(model.state_dict())

            total_loss = 0.0
            total_data = 0
            model.train()

            # print(torch.argmax(p1, dim=-1), torch.argmax(p2, dim=-1))
            # print(batch.s_idx, batch.e_idx)
    model.load_state_dict(best_model_params)
    return model

if __name__ == '__main__':
    data_config = DataConfig()
    data = DataProcessor(data_config)
    model_config = ModelConfig(data)

    model = BidafModel(model_config).to(model_config.device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=model_config.lr)
    loss_func = LogCriterion()
    model = train(model, data, optimizer, loss_func, model_config)