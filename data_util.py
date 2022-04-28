import torch
import numpy as np
import scipy.sparse as sp
from torchvision import datasets
from collections import namedtuple
from torchvision import datasets, transforms
import pickle as pk

def load_image(args):
    data_dir = "./data/" + str(args.dataset)

    data_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
    data_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

    trans = [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(0.1),
             transforms.RandomVerticalFlip(0.1),
             transforms.ToTensor(),
             transforms.Normalize(data_mean, data_std)]

    apply_transform = transforms.Compose(trans)

    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

    train_set.topk = 5
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)


    # split
    train_user_groups, test_user_groups, A = split_equal_noniid(
        train_set, test_set, args.shards, args.edge_frac, args.clients)


def load_cifar10(args):
    data_dir = "./data/" + str(args.dataset)

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)
    data_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
    data_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

    train_set.topk = 5
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    # split
    train_user_groups, test_user_groups, A = split_equal_noniid(
        train_set, test_set, args.shards, args.edge_frac, args.clients)

    train_set = list(zip(transpose(normalise(pad(train_set.data, 4), data_mean, data_std)), train_set.targets))
    test_set = list(zip(transpose(normalise(test_set.data, data_mean, data_std)), test_set.targets))

    train_batches = []
    test_batches = []
    for key, users in train_user_groups.items():
        train_batches.append(Batches(Transform([train_set[u.astype(int)] for u in users],
                train_transforms), args.batch_size, shuffle=True, device=args.device,
                set_random_choices=True, drop_last=True))
    for key, users in test_user_groups.items():
        test_batches.append(Batches([test_set[u.astype(int)] for u in users],
                args.batch_size, shuffle=False, device=args.device, drop_last=False))

    overall_tbatches = Batches(test_set, args.batch_size, shuffle=False,
                               device=args.device, drop_last=False)

    return train_batches, test_batches, A, overall_tbatches


# Image data related
def load_mnist(args):

    data_dir = "./data/" + str(args.dataset)

    trans = [transforms.ToTensor(),
         transforms.Normalize(*((0.1307,), (0.3081,)))]

    apply_transform = transforms.Compose(trans)

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    train_dataset.topk = 5
    train_dataset.data = torch.unsqueeze(train_dataset.data, 1)
    train_dataset.targets = np.array(train_dataset.targets)
    train_dataset.data = train_dataset.data.type(torch.FloatTensor)
    test_dataset.data = torch.unsqueeze(test_dataset.data, 1)
    test_dataset.targets = np.array(test_dataset.targets)
    test_dataset.data = test_dataset.data.type(torch.FloatTensor)

    train_user_groups, test_user_groups, A = split_equal_noniid(
        train_dataset, test_dataset, args.shards, args.edge_frac, args.clients)

    train_set = list(zip(train_dataset.data, train_dataset.targets))
    test_set = list(zip(test_dataset.data, test_dataset.targets))

    train_batches = []
    test_batches = []

    for key, users in train_user_groups.items():
        train_batches.append(Batches([train_set[u.astype(int)] for u in users], args.batch_size,
                                     shuffle=True, device=args.device, drop_last=True))

    for key, users in test_user_groups.items():
        test_batches.append(Batches([test_set[u.astype(int)] for u in users], args.batch_size,
                                    shuffle=False, device=args.device, drop_last=False))

    overall_tbatches = Batches(test_set, args.batch_size, shuffle=False,
                               device=args.device, drop_last=False)

    return train_batches, test_batches, A, overall_tbatches


def split_equal_noniid(train_dataset, test_dataset, shards, edge_frac, clients):
    """
    :param train_dataset:
    :param test_dataset:
    :param shards:
    :param edge_frac:
    :param clients:
    :return:
    """
    total_shards = shards * clients
    shard_size = int(len(train_dataset.data) / total_shards)
    idx_shard = [i for i in range(total_shards)]
    train_dict_users = {i: np.array([]) for i in range(clients)}
    idxs = np.arange(total_shards * shard_size)
    labels = train_dataset.targets
    dict_label_dist = {i: np.array([]) for i in range(clients)}

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_count = np.bincount(idxs_labels[1])

    # generate adj
    A = np.zeros((clients, clients))
    num_label = len(set(labels))
    label_dist = [[] for _ in range(num_label)]

    # partitions for train data
    for i in range(clients):
        rand_set = np.random.choice(idx_shard, shards, replace=False)
        idx_shard = list(set(idx_shard) - set(rand_set))
        selected_labels = idxs_labels[1, rand_set * shard_size]
        label_type = np.array(list(set(selected_labels)))
        sample_size = [np.count_nonzero(selected_labels == j) for j in label_type]
        int(shard_size * shards / len(label_type))
        dict_label_dist[i] = np.array((label_type, sample_size))

        for j, l in enumerate(label_type):
            start_idx = sum(label_count[0:l])
            end_idx = start_idx + label_count[l]
            sample_array = idxs[start_idx: end_idx]
            train_dict_users[i] = np.concatenate(
                (train_dict_users[i], np.random.choice(
                    sample_array, sample_size[j] * shard_size, replace=False)), axis=0)

        # for cifar-100, control the sparsity of A
        label_size = np.array([np.count_nonzero(
            labels[train_dict_users[i].astype(int)] == j) for j in label_type])
        pram_label_idx = np.array(sorted(range(len(label_size)),
                key=lambda i: label_size[i])[min(-train_dataset.topk, shards):])

        for label_type in label_type[pram_label_idx]:
            label_dist[label_type].append(i)

    # prepare A
    link_list = []
    for user_arr in label_dist:
        for user_a in user_arr:
            for user_b in user_arr:
                link_list.append([user_a, user_b])

    link_sample = list(range(len(link_list)))
    link_idx = np.random.choice(link_sample, int(edge_frac * len(link_list)), replace=False)
    for idx in link_idx:
        # A[link_list[idx][0], link_list[idx][1]] = A[link_list[idx][0], link_list[idx][1]] + 1
        A[link_list[idx][0], link_list[idx][1]] = 1

    # partition for test data
    total_shards = shards * clients
    shard_size = int(len(test_dataset.data) / total_shards)
    test_dict_users = {i: np.array([]) for i in range(clients)}
    idxs = np.arange(total_shards * shard_size)
    labels = test_dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_count = np.bincount(idxs_labels[1])

    for i in range(clients):
        for j, l in enumerate(dict_label_dist[i][0]):
            start_idx = sum(label_count[0:l])
            end_idx = start_idx + label_count[l]
            sample_array = idxs[start_idx: end_idx]
            test_dict_users[i] = np.concatenate(
                (test_dict_users[i], np.random.choice(
                    sample_array, dict_label_dist[i][1][j] * shard_size, replace=False)), axis=0)

    return train_dict_users, test_dict_users, torch.tensor(normalize_adj(A), dtype=torch.float32)


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
        if self.device is not None:
            return ({'input': x.to(self.device), 'target': y.to(self.device).long()} for (x, y) in self.dataloader)
        else:
            return ({'input': x, 'target': y.long()} for (x, y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


#####################
## data augmentation
#####################
class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:, y0:y0 + self.h, x0:x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)


class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}


class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:, y0:y0 + self.h, x0:x0 + self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}


class Transform:
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k, v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k: np.random.choice(v, size=N) for (k, v) in options.items()})


def normalise(x, mean, std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx