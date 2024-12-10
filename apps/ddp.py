import sys
import time
from random import Random
import numpy as np
from mpi4py import MPI

sys.path.append('./python')
import needle as ndl
import torch



def init():
    num_of_gpus = torch.cuda.device_count()
    global comm, world_size, rank, device
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    assert num_of_gpus >= world_size, f'Only {num_of_gpus} GPU(s) detected, but {world_size} GPU(s) required.'

    rank = comm.Get_rank()
    device = ndl.cuda(rank)
    print(f'Use cuda: {rank}')
    return rank, device


def partition_dataset(dataset, batch_size, device, dtype):
    bsz = batch_size // world_size
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(MPI.COMM_WORLD.Get_rank())
    print(f'partitioned dataset length: {len(partition)}')
    train_set = ndl.data.DataLoader(
        dataset=partition, batch_size=bsz, shuffle=True, device=device, dtype=dtype)
    return train_set, bsz


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def broadcast_parameters(model, root_rank=0):
    for p in model.parameters():
        p_data = p.numpy()
        p_data = comm.bcast(p_data, root=0)
        p.data = ndl.Tensor(p_data, device=device, dtype=p.dtype)


class DistributedOptimizer(ndl.optim.Optimizer):
    def __init__(self, opt):
        super().__init__(opt.params)
        self.opt = opt

    def step(self):
        self.average_gradients()
        self.opt.step()

    def average_gradients(self):
        for p in self.params:
            if p.grad is None:
                continue
            sendbuf = np.ascontiguousarray(p.grad.numpy())
            recvbuf = np.empty_like(sendbuf, dtype=p.dtype)
            comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
            original_grad=np.sum(sendbuf)
            recvbuf = recvbuf / world_size
            avg_grad=np.sum(recvbuf)
            print(f'original grad: {original_grad}; avg grad: {avg_grad}')
            p.grad.data = ndl.Tensor(recvbuf, device=device, dtype=p.grad.dtype)
