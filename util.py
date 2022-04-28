import datetime
import random
import os
import torch
import numpy as np
from collections import namedtuple
from functools import singledispatch


def print2file(buf, out_file, p=False):
    if p:
        print(buf)
    outfd = open(out_file, 'a+')
    outfd.write(str(datetime.datetime.now()) + '\t' + buf + '\n')
    outfd.close()


def initial_environment(seed, cpu_num=5, deterministic=False):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sd_matrixing(state_dic):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    keys = []
    param_vector = None
    for key, param in state_dic.items():
        keys.append(key)
        if param_vector is None:
            param_vector = param.clone().detach().flatten().cpu()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
            else:
                param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
    return param_vector


def trainable_params(model):
    result = []
    for p in model.parameters():
        if p.requires_grad:
            result.append(p)
    return result


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


class StatsLogger():
    def __init__(self, keys):
        self._stats = {k: [] for k in keys}

    def append(self, output):
        for k, v in self._stats.items():
            v.append(output[k].detach())

    def stats(self, key):
        return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=np.float)


@singledispatch
def cat(*xs):
    raise NotImplementedError


@singledispatch
def to_numpy(x):
    raise NotImplementedError


@cat.register(torch.Tensor)
def _(*xs):
    return torch.cat(xs)


@to_numpy.register(torch.Tensor)
def _(x):
    return x.detach().cpu().numpy()