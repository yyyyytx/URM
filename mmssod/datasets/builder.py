# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
import warnings
from functools import partial

import numpy as np
# from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader
from mmdet.datasets.builder import worker_init_fn
# from mmdet.datasets.samplers import (DistributedGroupSampler, DistributedSampler,
#                        GroupSampler, InfiniteBatchSampler,
#                        InfiniteGroupBatchSampler)
from .sampler import *
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from mmcv.parallel import DataContainer


DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_semi_dataloader(dataset,
                          sampler_ratio,
                          samples_per_gpu,
                          workers_per_gpu,
                          epoch_length=1036,
                          num_gpus=1,
                          dist=True,
                          shuffle=True,
                          seed=None,
                          runner_type='EpochBasedRunner',
                          persistent_workers=False,
                          **kwargs):

    rank, world_size = get_dist_info()
    assert len(sampler_ratio) == 2
    assert samples_per_gpu % (sampler_ratio[0] + sampler_ratio[1]) == 0

    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    # batch_sampler = DistributedSemiBatchSampler(dataset=dataset,
    #                                             sample_ratio=sampler_ratio,
    #                                             samples_per_gpu=samples_per_gpu,
    #                                             epoch_length=epoch_length,
    #                                             seed=seed)
    batch_sampler = DistributedInfiniteSemiGroupBatchSampler(dataset=dataset,
                                                             sample_ratio=sampler_ratio,
                                                             samples_per_gpu=samples_per_gpu)
    sampler = None
    batch_size = 1

    # sampler = GroupSampler(dataset,
    #                        samples_per_gpu) if shuffle else None
    #
    # batch_sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if (TORCH_VERSION != 'parrots'
            and digit_version(TORCH_VERSION) >= digit_version('1.7.0')):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            collate, samples_per_gpu=samples_per_gpu, flatten=True),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def collate(batch, samples_per_gpu=1, flatten=False):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i: i + samples_per_gpu]]
                )
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True
            )
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i: i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(
                                max_shape[dim - 1], sample.size(-dim)
                            )
                    padded_samples = []
                    for sample in batch[i: i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - \
                                sample.size(-dim)
                        padded_samples.append(
                            F.pad(sample.data, pad, value=sample.padding_value)
                        )
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate(
                            [sample.data for sample in batch[i: i + samples_per_gpu]]
                        )
                    )
                else:
                    raise ValueError(
                        "pad_dims should be either None or integers (1-3)")

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i: i + samples_per_gpu]]
                )
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif any([isinstance(b, Sequence) for b in batch]):
        if flatten:
            flattened = []
            for b in batch:
                if isinstance(b, Sequence):
                    flattened.extend(b)
                else:
                    flattened.extend([b])
            return collate(flattened, len(flattened))
        else:
            transposed = zip(*batch)
            return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu) for key in batch[0]
        }
    else:
        return default_collate(batch)
