from __future__ import division

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler, WeightedRandomSampler, BatchSampler
import math
import itertools
import random

class DistributedSemiBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 sample_ratio=None,
                 samples_per_gpu=1,
                 epoch_length=7330,
                 world_size=None,
                 rank=None,
                 seed=0,
                 shuffle=True):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank


        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.batch_size = samples_per_gpu
        self.seed = seed if seed is not None else 0
        self.shuffle = shuffle
        self.cumulative_sizes = dataset.cumulative_sizes
        self.epoch_length = epoch_length
        self.n_sample_label = int(samples_per_gpu / (sample_ratio[0] + sample_ratio[1]) * sample_ratio[0])
        self.n_sample_unlabel = int(samples_per_gpu / (sample_ratio[0] + sample_ratio[1]) * sample_ratio[1])

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.epoch = 0
        self.size = len(dataset)
        # self._gen_indices_of_label()
        # self._gen_indices_of_unlabel()


    def _gen_indices_of_label(self):
        self.flag += 1
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed + self.flag)
        indices = torch.randperm(self.cumulative_sizes[0], generator=g).tolist()
        return iter(indices[self.rank::self.world_size])

    def _gen_indices_of_unlabel(self):
        self.flag += 1
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed + self.flag)
        indices = torch.randperm(self.cumulative_sizes[1]-self.cumulative_sizes[0], generator=g).tolist()
        return iter(indices[self.rank::self.world_size])


    def __iter__(self):
        print('create indices')
        self.flag = 0
        label_indices = self._gen_indices_of_label()
        unlabel_indices = self._gen_indices_of_unlabel()
        # once batch size is reached  yield the indices
        # while True:
        for i in range(self.epoch_length):
            batch_buffer = []
            for _ in range(self.n_sample_label):
                try:
                    idx_l = next(label_indices)
                    batch_buffer.append(idx_l)
                except:
                    print('re-gen label index')
                    label_indices = self._gen_indices_of_label()
                    idx_l = next(label_indices)
                    batch_buffer.append(idx_l)
            for _ in range(self.n_sample_unlabel):
                try:
                    idx_u = next(unlabel_indices)
                    batch_buffer.append(idx_u + self.cumulative_sizes[0])
                except:
                    print('re-gen unlabel index')
                    unlabel_indices = self._gen_indices_of_unlabel()
                    idx_u = next(unlabel_indices)
                    batch_buffer.append(idx_u + self.cumulative_sizes[0])
            yield batch_buffer

    def __len__(self):
        """Length of base dataset."""
        return self.epoch_length

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSemiGroupBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 sample_ratio=None,
                 samples_per_gpu=1,
                 world_size=None,
                 rank=None,
                 seed=0):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank


        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        assert hasattr(self.dataset, 'flag')

        self.sample_ratio = sample_ratio
        assert samples_per_gpu % (self.sample_ratio[0] + sample_ratio[1]) == 0
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        # self.buffer_per_group = {k: [] for k in range(len(self.group_sizes))}
        # print("group size:", self.group_sizes)
        # print("buffer per group:", self.buffer_per_group)

        self.labeled_flag = self.dataset.labeled.flag
        self.labled_group_sizes = np.bincount(self.labeled_flag)
        self.unlabeled_flag = self.dataset.unlabeled.flag
        self.unlabeled_group_sizes = np.bincount(self.unlabeled_flag)
        # print("labeled:", self.labled_group_sizes)
        # print("unlabeled:", self.unlabeled_group_sizes)

        self.batch_size = samples_per_gpu
        self.seed = seed if seed is not None else 0
        self.cumulative_sizes = dataset.cumulative_sizes


        self.indice_batches = self._gen_semi_batch()
        # print("before:", self.indice_batches[:10])
        random.shuffle(self.indice_batches)
        # print("after:", self.indice_batches[:10])


    def _gen_semi_batch(self):
        np.random.seed(self.seed)
        indice_batches = []
        for group_idx in range(len(self.group_sizes)):
            labeled_indice = np.where(self.labeled_flag == group_idx)[0]
            unlabeled_indice = np.where(self.unlabeled_flag == group_idx)[0] + self.cumulative_sizes[0]

            shuffled_labeled_indice = []
            shuffled_unlabeled_indice = []

            labeled_times = len(labeled_indice) // self.sample_ratio[0]
            unlabeled_times = len(unlabeled_indice) // self.sample_ratio[1]
            if labeled_times > unlabeled_times:
                np.random.shuffle(labeled_indice)
                shuffled_labeled_indice.extend(labeled_indice)

                unlabeled_repeat_times = labeled_times // unlabeled_times + 1
                for i in range(unlabeled_repeat_times):
                    np.random.shuffle(unlabeled_indice)
                    shuffled_unlabeled_indice.extend(unlabeled_indice)

                for i in range(labeled_times):
                    tmp_buffer = []
                    tmp_buffer.extend(shuffled_labeled_indice[i*self.sample_ratio[0]:(i+1)*self.sample_ratio[0]])
                    tmp_buffer.extend(shuffled_unlabeled_indice[i*self.sample_ratio[1]:(i+1)*self.sample_ratio[1]])
                    indice_batches.append(tmp_buffer)
            else:
                labeled_repeat_times = unlabeled_times // labeled_times + 1
                for i in range(labeled_repeat_times):
                    np.random.shuffle(labeled_indice)
                    shuffled_labeled_indice.extend(labeled_indice)

                np.random.shuffle(unlabeled_indice)
                shuffled_unlabeled_indice.extend(unlabeled_indice)

                for i in range(unlabeled_times):
                    tmp_buffer = []
                    tmp_buffer.extend(shuffled_labeled_indice[i*self.sample_ratio[0]:(i+1)*self.sample_ratio[0]])
                    tmp_buffer.extend(shuffled_unlabeled_indice[i*self.sample_ratio[1]:(i+1)*self.sample_ratio[1]])
                    indice_batches.append(tmp_buffer)

        # print("indice_batches:", indice_batches[:10])
        last_count = len(indice_batches) % self.world_size
        if last_count != 0:
            remain_count = self.world_size - last_count
            for i in range(remain_count):
                indice_batches.append(indice_batches[i])

        return indice_batches[self.rank::self.world_size]

    def __iter__(self):
        for batch_item in self.indice_batches:
            yield batch_item

    def __len__(self):
        """Length of base dataset."""
        return len(self.indice_batches)

    def set_epoch(self, epoch):
        self.epoch = epoch

