import sys
import time
import numpy as np
from mpi4py import MPI
sys.path.append('./python')
sys.path.append('./apps')
from model_parallel import ModelParallelResNet9,init_model_parallel
import needle as ndl
import ddp

if __name__ == "__main__":
    np.random.seed(0)
    rank, device = ddp.init()
    
    # Initialize dataset on all GPUs
    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(dataset, 128, device=device, dtype="float32")

    print(f'dataset length: {len(dataset)}')

    num_gpus, comm, world_size, rank, device = init_model_parallel()
    model = ModelParallelResNet9(num_gpus, device=device, dtype="float32")
    # Initialize optimizer and loss function on both first and last GPU
    loss_fn = ndl.nn.SoftmaxLoss()

    opt = ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    
    begin = time.time()
    for epoch in range(1):
        print(f'rank: {rank}, epoch: {epoch+1}')
        
        for i, batch in enumerate(train_loader):
            # Reset gradients on all GPUs
            if rank == 0:
                opt.reset_grad()
            
            # Broadcast input data from GPU 0 to all GPUs
            if rank == 0:
                X, y = batch
                X_data = X.numpy()
                y_data = y.numpy()
            else:
                X_data = None
                y_data = None

            # Synchronize before broadcast
            comm.Barrier()
            X_data = comm.bcast(X_data, root=0)
            y_data = comm.bcast(y_data, root=0)

            # Create tensors on each GPU
            X = ndl.Tensor(X_data, device=device)
            y = ndl.Tensor(y_data, device=device)

            # Forward pass (already handles GPU communication)
            out = model(X)
            # Synchronize after foward pass
            comm.Barrier()

            # Backward pass
            if rank == 3:
                # Only compute loss on the last GPU
                loss = loss_fn(out, y)
                loss.backward()
                model.backward(out.grad)
            else:
                # Other GPUs wait for gradient
                #print(f'rank: {rank} get gradients')
                grad_output = ndl.Tensor(np.zeros_like(out.numpy()), device=device)
                model.backward(grad_output)

            # Synchronize after backward pass
            comm.Barrier()

            opt.step()

            comm.Barrier()

            # Print metrics
            if rank == 3 and i % 100 == 0:
                correct = np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
                acc = correct / y.shape[0]
                print(f'batch {i}, acc: {acc:.4f}, loss: {loss.numpy():.4f}')

    if rank == 0:
        end = time.time()
        print(f'Training Time: {end-begin}')   
