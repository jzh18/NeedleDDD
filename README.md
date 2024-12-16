# DDDML
Authors:

* Huaifeng Zhang, Chalmers University of Technology

## Introduction
NeedleDDD is a machine learning framework comparable to TensorFlow and PyTorch, offering a comprehensive set of libraries for building, training, and deploying deep learning models. 
It can be used both on CPUs and GPUs.

The framework supports three parallelism strategies: data parallelism, model parallelism, and pipeline parallelism. 

Additionally, NeedleDDD includes an experimental feature for decentralized computing. Currently, this feature is limited to specific operations, such as `matmul`.
The decentralization feature is mostly inspired by the Ethereum blockchain, and it extends the functionality of the `geth` client to create a decentralized network.
Please refer to https://github.com/jzh18/go-ethereum for more information about the extended `geth` client.


## Documents for the Presentation
* [Presentation Doc](https://github.com/jzh18/NeedleDDD/blob/main/final_project_report.ipynb)

## Contributions
All the authors engaged in in-depth discussions on implementing decentralized features, particularly focusing on verifying the nodes working honestly.
And different approaches were proposed to achieve this goal.
Here is the detailed contribution of each author:
* Huaifeng Zhang: Found the matrix multiplication verification algorithm and implemented the decentralized computing feature.