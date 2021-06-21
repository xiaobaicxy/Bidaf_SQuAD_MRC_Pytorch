# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class LogCriterion(nn.Module):
    def __init__(self):
        super(LogCriterion, self).__init__()
        
    def forward(self, predicts, targets):
        # predicts = (p1, p2)
        # p1, p2: [batch_size, seq_len]
        # targets = (s_idx, e_idx)
        # s_idx, e_idx: [batch_size]
        p1, p2 = predicts
        s_idx, e_idx = targets

        batch_size = s_idx.size(0)
        p_y1 = torch.stack([p1[i, s_idx[i]] for i in range(batch_size)], dim=0).view(-1) # [batch_size]
        p_y2 = torch.stack([p2[i, e_idx[i]] for i in range(batch_size)], dim=0).view(-1)

        loss_start = -torch.sum(torch.log(p_y1))
        loss_end = -torch.sum(torch.log(p_y2))
        loss = loss_start + loss_end

        return loss