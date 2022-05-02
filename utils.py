import torch
import torch.nn as nn


def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    return net


def PSNR_batch(true, pred):
    max_value = torch.tensor([1.])
    psnr = 10 * torch.log10((max_value ** 2) / torch.mean(torch.pow((true - pred), 2)))
    return psnr


def load_state_dict_cpu(net, state_dict0):
    state_dict1 = net.state_dict()
    for name, value in state_dict1.items():
        assert 'module.'+name in state_dict0
        state_dict1[name] = state_dict0['module.'+name].cpu()
    net.load_state_dict(state_dict1)
