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
    
    num_microbatches = 4  # Number of microbatches in pipeline

    # Initialize dataset on all GPUs
    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(dataset, 128*num_microbatches, device=device, dtype="float32")

    print(f'dataset length: {len(dataset)}')

    num_gpus, comm, world_size, rank, device = init_model_parallel()
    model = ModelParallelResNet9(num_gpus, device=device, dtype="float32")
    # Initialize optimizer and loss function on both first and last GPU
    loss_fn = ndl.nn.SoftmaxLoss()

    opt = ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    
    begin = time.time()
    
    for i, batch in enumerate(train_loader):
        # Split batch into microbatches
        X, y = batch
        # Broadcast input data from GPU 0 to all GPUs
        if rank == 0:
            X_data = X.numpy()
            y_data = y.numpy()
        else:
            X_data = None
            y_data = None

        # Synchronize before broadcast
        comm.Barrier()

        # Broadcast the entire batch data to all GPUs
        X_data = comm.bcast(X_data, root=0)
        y_data = comm.bcast(y_data, root=0)

        # Create tensors on each GPU
        X = ndl.Tensor(X_data, device=device)
        y = ndl.Tensor(y_data, device=device)

        # Convert tensors to NumPy arrays for slicing
        X_np = X.numpy()
        y_np = y.numpy()

        # Calculate the size of each microbatch
        batch_size = X_np.shape[0]
        microbatch_size = batch_size // num_microbatches

        # Manually split the data into microbatches using NumPy slicing
        microbatches_X = [ndl.Tensor(X_np[i * microbatch_size:(i + 1) * microbatch_size], device=device) for i in range(num_microbatches)]
        microbatches_y = [ndl.Tensor(y_np[i * microbatch_size:(i + 1) * microbatch_size], device=device) for i in range(num_microbatches)]

        # Handle any remaining data if batch_size is not perfectly divisible
        if batch_size % num_microbatches != 0:
            microbatches_X[-1] = ndl.Tensor(X_np[(num_microbatches - 1) * microbatch_size:], device=device)
            microbatches_y[-1] = ndl.Tensor(y_np[(num_microbatches - 1) * microbatch_size:], device=device)

        # Now microbatches_X and microbatches_y contain the split data
        # Each element in these lists corresponds to a microbatch
            
        # Dictionary to store outputs of the forward pass
        forward_outputs = {}

        # Forward pass loop
        for step in range(num_microbatches):
            X_mb = microbatches_X[step]
            out = model(X_mb)
            forward_outputs[step] = out  # Store the output with step as key

            out = forward_outputs.get(step)  # Retrieve the output for this step
            if out is not None:
                if rank == 3:
                    loss = loss_fn(out, microbatches_y[step])
                    loss.backward()
                    if (i*num_microbatches)% 100 == 0:
                        # Calculate accuracy
                        correct = np.sum(np.argmax(out.numpy(), axis=1) == microbatches_y[step].numpy())
                        acc = correct / microbatches_y[step].shape[0]
                        # Print metrics
                        print(f'Batch {i*num_microbatches}, Microbatch {step}, Acc: {acc:.4f}, Loss: {loss.numpy():.4f}')

                    model.backward(out.grad)
                else:
                    grad_output = ndl.Tensor(np.zeros_like(out.numpy()), device=device)
                    model.backward(grad_output)

                # Perform optimizer step for each microbatch
                opt.step()

        # Ensure optimizer step is called after all microbatches are processed
        comm.Barrier()

    if rank == 0:
        end = time.time()
        print(f'Training Time: {end-begin}')            
