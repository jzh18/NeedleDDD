import sys
import time
from random import Random
import numpy as np
from mpi4py import MPI
sys.path.append('./python')
sys.path.append('./apps')
from simple_training import train_cifar10, evaluate_cifar10
from models import ResNet9
import needle as ndl
import ddp

if __name__ == "__main__":
    np.random.seed(0)
    rank, device = ddp.init()

    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)

    train_set, bsz = ddp.partition_dataset(
        dataset, 128, device=device, dtype='float32')
    print(f'orignal dataset length: {len(dataset)}')

    model = ResNet9(device=device, dtype="float32")
    ddp.broadcast_parameters(model)

    model.train()
    correct, total_loss = 0, 0
    opt = ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    opt = ddp.DistributedOptimizer(opt)
    loss_fn = ndl.nn.SoftmaxLoss()
    n_epochs = 1
    begin = time.time()
    for i in range(n_epochs):
        if rank == 0:
            print(f'epoch: {i+1}/{n_epochs}')
        count = 0
        for batch in train_set:
            opt.reset_grad()
            X, y = batch
            out = model(X)
            correct = np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            acc = correct/(y.shape[0])
            if rank == 0 and count % 100 == 0:
                print(f'acc: {acc}; avg_loss: {loss.data.numpy()}')
            count += 1
    end = time.time()
    if rank == 0:
        print(f'Training Time: {end-begin}')
