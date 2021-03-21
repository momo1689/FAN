import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self, loss_mode):
        super(Loss, self).__init__()
        self.loss_mode = loss_mode
        self.gamma = 0.2

    def forward(self, pred_list, gt_list):
        pred_img = pred_list[0]
        pred_map = pred_list[1]
        gt_img = gt_list[0]
        gt_map = gt_list[1]

        if self.loss_mode == 'l1_map':
            map_loss = torch.mean(abs(pred_map - gt_map))
            img_loss = torch.mean(abs(pred_img - gt_img))
            loss = img_loss + self.gamma * map_loss
        elif self.loss_mode == 'l2_map':
            map_loss = torch.mean(abs(pred_map - gt_map))
            img_loss = torch.mean(torch.pow((pred_img - gt_img), 2))
            loss = img_loss + self.gamma * map_loss
        elif self.loss_mode == 'l1':
            img_loss = torch.mean(abs(pred_img - gt_img))
            loss = img_loss
        elif self.loss_mode == 'l2':
            img_loss = torch.mean(torch.pow((pred_img - gt_img), 2))
            loss = img_loss
        else:
            raise ValueError('Wrong loss mode, please choose l1_map, l2_map, l1 or l2')
        return loss
