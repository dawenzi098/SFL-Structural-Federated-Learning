import torch.nn as nn
import torch
from collections import namedtuple
import numpy as np

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}
sep = '_'
RelativePath = namedtuple('RelativePath', ('parts'))
rel_path = lambda *parts: RelativePath(parts)


class BResidual(nn.Module):
    def __init__(self, reg_channel):
        losses = {
            'loss': (nn.CrossEntropyLoss(reduction='none'), [('classifier',), ('target',)]),
            'correct': (Correct(), [('classifier',), ('target',)]),
        }
        network = union(net(reg_channel), losses)

        self.graph = build_graph(network)
        super().__init__()
        for n, (v, _) in self.graph.items():
            setattr(self, n, v)

    def forward(self, inputs):
        self.cache = dict(inputs)
        for n, (_, i) in self.graph.items():
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache

    def half(self):
        # for module in self.children():
        #     if type(module) is not nn.BatchNorm2d:
        #         module.half()
        return self

    def get_state(self, mode="full"):
        # return [i for i in self.named_parameters()], []
        return self.state_dict(), []

    def set_state(self, w_server, w_local, mode="full"):
        # sd = self.state_dict()
        # for key, param in w_server:
        #     if key in sd.keys():
        #         sd[key] = param.clone().detach()
        #     else:
        #         print("Server layers mismatch at 'set_state' function.")
        #
        # for key, param in w_local:
        #     if key in sd.keys():
        #         sd[key] = param.clone().detach()
        #     else:
        #         print("Local layers mismatch at 'set_state' function.")

        self.load_state_dict(w_server)


def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        'relu': nn.ReLU(True)
    }


def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }


def basic_net(channels, weight,  pool, **kw):
    return {
        'prep': conv_bn(channels["reg"], channels['prep'], **kw),
        # 'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'pool': nn.MaxPool2d(8),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], 10, bias=False),
        'classifier': Mul(weight),
    }


def net(reg_channel, channels=None, weight=0.2, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer2'), **kw):
    channels = channels or {'reg': reg_channel, 'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 256, }
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)
    return n


def build_graph(net):
    net = dict(path_iter(net))
    default_inputs = [[('input',)]]+[[k] for k in net.keys()]
    with_default_inputs = lambda vals: (val if isinstance(val, tuple) else (val, default_inputs[idx]) for idx, val in enumerate(vals))
    parts = lambda path, pfx: tuple(pfx) + path.parts if isinstance(path, RelativePath) else (path,) if isinstance(path, str) else path
    return {sep.join((*pfx, name)): (val, [sep.join(parts(x, pfx)) for x in inputs]) for (*pfx, name), (val, inputs) in zip(net.keys(), with_default_inputs(net.values()))}


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m


class Identity(nn.Module):
    def forward(self, x): return x


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))


class Add(nn.Module):
    def forward(self, x, y): return x + y


class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target


#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, device, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.device = device
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle,
            drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        # return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x, y) in self.dataloader)
        if self.device is not None:
            return ({'input': x.to(self.device), 'target': y.to(self.device).long()} for (x, y) in self.dataloader)
        else:
            return ({'input': x, 'target': y.long()} for (x, y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)
