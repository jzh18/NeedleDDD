# NeedleDDD: A Distributed and Decentralized Machine Learning Framework
Authors:

* Huaifeng Zhang, Chalmers University of Technology
* Qi Shao, Chalmers University of Technology
* Hantang Zhang, Umeå University
* Eric Olsson, Chalmers University of Technology

## Introduction
NeedleDDD is a machine learning framework comparable to TensorFlow and PyTorch, offering a comprehensive set of libraries for building, training, and deploying deep learning models.
The framework is derived from the CMU 10-414/714: Deep Learning Systems course.
It can be used both on CPUs and GPUs.

The framework supports three parallelism strategies: data parallelism, model parallelism, and pipeline parallelism.

Additionally, NeedleDDD includes an experimental feature for decentralized computing. Currently, this feature is limited to specific operations, such as `matmul`.
The decentralization feature is mostly inspired by the Ethereum blockchain, and it extends the functionality of the `geth` client to create a decentralized network.
Please refer to https://github.com/jzh18/go-ethereum for more information about the extended `geth` client.

A sketch of the decentralized computing system has been analyzed from a security perspective.

## Quick Start
1. Install dependencies: `!pip3 install pybind11`
2. Build C and CUDA backend: `make`
3. Run unit tests to verify the framework is working correctly: `!python3 -m pytest -l -v -k "nd_backend"`. This should ouput `PASSED` for all tests.

## Distributed ML and Decentralized ML
Check the [Presentation Doc](https://github.com/jzh18/NeedleDDD/blob/main/final_project_report.ipynb) for more information.

## Documents for the Presentation
* [Presentation Doc](https://github.com/jzh18/NeedleDDD/blob/main/final_project_report.ipynb)

## Contributions
All the authors engaged in in-depth discussions on implementing decentralized features, particularly focusing on verifying the nodes working honestly.
And different approaches were proposed to achieve this goal.
Here is the detailed contribution of each author:
* Huaifeng Zhang:
    * Huaifeng Zhang found the [Freivalds' algorithm](https://en.wikipedia.org/wiki/Freivalds%27_algorithm) to verify the matrix multiplication results. He also designed the architecture of the decentralized network. He implemented the decentralized `matmul` operation in the NeedleDDD framework. The `matmul` operation is packaged as a Transaction and submitted to a decentralized network to be executed by the node in the network. He also developed a client based on the [Geth](https://github.com/jzh18/go-ethereum) client to create a decentralized network. Specifically, he changed the consensus mechanism of the Geth client from a Proof of Work (PoW) to a Proof of Useful Work (PoUW) mechanism. The nodes in the network are rewarded based on the number of correct results they provide. The client can also verify the results of the nodes using the Freivalds' algorithm.

* Hantang Zhang:
   * Hantang gained a comprehensive understanding of the overall project workflow. He conducted an in-depth analysis of the autodiff and GEMM algorithms, identifying their optimal use cases in the project. He wrote the "Introduce Autodiff" and "Introduce GEMM" sections to explain their roles and implementation details.

* Eric Olsson:
    * Eric investigated how to achieve the project goal of decentralized machine learning, while retaining sufficient privacy and security. He formulated a threat model for the proposed decentralized needle scheme. He then surveyed relevant work in decentralized data marketplaces, federated learning, and machine learning. This survey found OmniLytics, a highly relevant blockchain-based secure data trading marketplace. Eric finally highlighted remaining flaws within the threat model, and future work. This work is described in the 'Decentralized S&P Analysis' section.
 
* Qi Shao:
    * For large models that gpu cannot support the allocation of memory for training, model parallel training is one of the solution. Model parallel trainning will spilt different layers of neural network on to different gpu to reduce memory cost. He implemented model parallel training, split model layers into different gpu and pass intermedidate forward and backward result.
    * Model parallel traning might introduce the cost of waiting time.Take an example, GPU1 has to wait GPU0 to finish its compute task of computing specific layers for all the data. Datapipeline model parapllel training spilt data into microbatches to reduce the waiting time.After GPU0 finishes the first microbatch of computing, it can forward data to GPU1, which reduces time. He implemented datapipeline model parallel training, spilt data into batches to flow in pipeline and work together with model parallel training. Try to improve the performance.
