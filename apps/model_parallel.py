import sys
import time
from mpi4py import MPI
import numpy as np
import needle as ndl
from models import ConvBN
from mpi4py import MPI
import torch

def init_model_parallel():
    """Initialize MPI environment for model parallel training"""
    num_gpus = torch.cuda.device_count()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    
    assert num_gpus >= world_size, f'Only {num_gpus} GPU(s) detected, but {world_size} GPU(s) required.'
    
    device = ndl.cuda(rank)
    return num_gpus, comm, world_size, rank, device


class ModelParallelResNet9(ndl.nn.Module):
    """ResNet9 split across multiple GPUs"""

    def __init__(self, num_gpus, device=None, dtype="float32"):
        super().__init__()
        self.num_gpus = num_gpus
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        if self.rank == 0:
            # First GPU: Initial convolutions
            self.cb1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
            self.cb2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)

        elif self.rank == 1:
            # Second GPU: First residual block
            self.cb3 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
            self.cb4 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)

        elif self.rank == 2:
            # Third GPU: Second set of convolutions
            self.cb5 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
            self.cb6 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)

        elif self.rank == 3:
            # Fourth GPU: Final residual block and classifier
            self.cb7 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
            self.cb8 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
            self.linear1 = ndl.nn.Linear(128, 128, device=device, dtype=dtype)
            self.relu = ndl.nn.ReLU()
            self.linear2 = ndl.nn.Linear(128, 10, device=device, dtype=dtype)

    def forward(self, x):
        # Store expect_grad_shapes for backward pass
        if self.rank == 0:
            x = self.cb1(x)
            x = self.cb2(x)
            self.comm.send(x.numpy(), dest=1)
            self.saved_for_backward = x
            self.expect_grad_shape = x.shape

        elif self.rank == 1:
            x = self.comm.recv(source=0)
            x = ndl.Tensor(x, device=ndl.cuda(self.rank))
            x_residual = x
            x = self.cb3(x)
            x = self.cb4(x)
            x = x + x_residual
            self.comm.send(x.numpy(), dest=2)
            self.saved_for_backward = (x, x_residual)
            self.expect_grad_shape = x.shape

        elif self.rank == 2:
            x = self.comm.recv(source=1)
            x = ndl.Tensor(x, device=ndl.cuda(self.rank))
            input_cb5 = x
            x = self.cb5(x)
            input_cb6 = x
            x = self.cb6(x)
            self.comm.send(x.numpy(), dest=3)
            self.saved_for_backward = (input_cb5, input_cb6, x)
            self.expect_grad_shape = x.shape

        elif self.rank == 3:
            x = self.comm.recv(source=2)
            x = ndl.Tensor(x, device=ndl.cuda(self.rank))
            x_residual = x
            x = self.cb7(x)
            x = self.cb8(x)
            x = x + x_residual
            N, C, H, W = x.shape
            x = x.reshape((N, C * H * W))
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            self.saved_for_backward = (x, x_residual)
            self.expect_grad_shape = x.shape

        return x

    def backward(self, grad_output):
        """Handle gradient communication between GPUs"""
        if self.rank == 3:
            self.saved_for_backward[0].backward(grad_output)
            grad = self.saved_for_backward[1].grad
            self.comm.send(grad.numpy(), dest=2)

        elif self.rank == 2:
            grad = self.comm.recv(source=3)
            grad = ndl.Tensor(grad, device=ndl.cuda(self.rank))
            
            input_cb5, input_cb6, output = self.saved_for_backward
            output.backward(grad)  
            grad = input_cb6.grad         

            x = self.cb5(input_cb5)  
            x.backward(grad)         
            grad = input_cb5.grad    
            
            self.comm.send(grad.numpy(), dest=1)

        elif self.rank == 1:
            grad = self.comm.recv(source=2)
            grad = ndl.Tensor(grad, device=ndl.cuda(self.rank))
            
            output, x_residual = self.saved_for_backward
            grad = grad + x_residual.grad if hasattr(x_residual, 'grad') else grad
            
            x = self.cb4(x_residual)
            x.backward(grad)
            grad = x_residual.grad
            
            x = self.cb3(x_residual)
            x.backward(grad)
            grad = x_residual.grad
            
            self.comm.send(grad.numpy(), dest=0)

        elif self.rank == 0:
            grad = self.comm.recv(source=1)
            grad = ndl.Tensor(grad, device=ndl.cuda(self.rank))
            self.saved_for_backward.backward(grad)

