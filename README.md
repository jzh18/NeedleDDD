# DDDML
Authors:

* Huaifeng Zhang, Chalmers University of Technology
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
