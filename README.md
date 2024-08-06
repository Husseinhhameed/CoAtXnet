# CoAtXNet: Cross-Attention Strategy for Utilizing RGB-D Images for Camera Localization

## Overview

CoAtXNet is a hybrid model that leverages the strengths of both Convolutional Neural Networks (CNNs) and Transformers to enhance vision-based camera localization. By integrating RGB and depth images through cross-attention mechanisms, CoAtXNet significantly improves feature representation and bidirectional information flow between modalities. This approach combines the local feature extraction capabilities of CNNs with the global context modeling strengths of Transformers, resulting in superior performance across various indoor scenes.
![CoAtXNet Architecture](https://github.com/Husseinhhameed/CoAtXnet/blob/main/Images/Architecture.png)
## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Implementation Details](#implementation-details)
- [Experiments](#experiments)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [Contact](#contact)

## Introduction
Camera localization, the process of determining a camera’s position and orientation within an environment, plays a pivotal role in the functionality of several systems. Traditional localization methods, including those based on structure and convolutional neural networks (CNNs), often encounter limitations in dynamic or visually complex environments.

This repository contains the implementation of CoAtXNet, a novel hybrid architecture that merges CNNs and Transformers using cross-attention mechanisms to efficiently integrate RGB and depth images. CoAtXNet processes these modalities independently through convolutional layers and then combines them with cross-attention, resulting in enhanced feature representation.

## Methodology
CoAtXNet utilizes dual streams to process RGB and depth information independently using convolutional layers and then fuses these features with cross-attention mechanisms. This design leverages the detailed texture information from RGB images and geometric depth cues to enhance localization accuracy.

### Key Contributions
- **Cross-Attention Mechanism:** Novel cross-attention mechanisms fuse features from the RGB and depth streams, helping the model better grasp local and global contexts.
- **Dual-Stream Hybrid Architecture:** A new dual-stream version of the hybrid CNN-transformer network processes the RGB and depth images through convolutional layers separately and then combines them using transformer-based cross-attention, optimizing the strengths of both approaches.

## Implementation Details
We implemented our proposed model using PyTorch, leveraging the Adam optimizer with an initial learning rate of 0.0001 and a batch size of 32. Input images were resized to a size of 256 × 256 pixels. During both training and testing, preprocessing applied to the images includes resizing and normalization into tensors.

The network was trained using the K-fold cross-validation method with 5 splits. Each fold was trained for 150 epochs. The learning rate was dynamically adjusted using a ReduceLROnPlateau scheduler that drops it by a factor of 0.1 if the validation loss does not improve over 10 epochs.

## Experiments
In this work, we utilize the 7Scenes dataset, a well-known benchmark to evaluate vision-based camera localization. It includes a collection of seven different indoor scenes captured with a handheld Kinect RGB-D camera device.

## Results
The results demonstrate that all variants of the CoAtXNet model achieve competitive performance across different scenes, with CoAtXNet-4 showing the best overall accuracy in terms of both translation and orientation errors.
![CoAtXNet Architecture](https://github.com/Husseinhhameed/CoAtXnet/blob/main/Images/result.png)
![CoAtXNet Architecture](https://github.com/Husseinhhameed/CoAtXnet/blob/main/Images/result2.png)

## Discussion
The experimental results highlight the superior performance of the CoAtXNet model in the domain of absolute pose regression. By combining the strengths of traditional Convolutional Neural Networks (CNNs) with transformers, CoAtXNet effectively utilizes both local and global features, leading to improvements in position and orientation accuracy.

## Conclusion
CoAtXNet represents a substantial advancement in the field of camera localization by effectively combining CNNs and Transformers through cross-attention mechanisms. This work not only enhances the accuracy and robustness of camera localization but also opens new avenues for research in hybrid models for various vision-based tasks.

## How to Use


1. To run the implementation on Google Colab, open the provided `CoAtXNet.ipynb` notebook:
    - Open [Google Colab](https://colab.research.google.com/)
    - Upload the `CoAtXNet.ipynb` notebook
    - Follow the instructions in the notebook to run the complete implementation

## Repository Contents
- `CoAtXNet.ipynb`: Jupyter notebook for running the complete implementation on Google Colab.
- `requirements.txt`: List of dependencies required to run the code.
- `Trainig.py`: Script to train the CoAtXNet model.
- `Model.py`: Script to define the CoAtXNet model.
- `LoadData.py`: Script to load and prepocess data from 7sence dataset.

## Acknowledgements
We used the CoAtNet implementation from [CoAtNet PyTorch](https://github.com/Husseinhhameed/coatnet-pytorch).

## Contact
If you have any questions or need further assistance, please feel free to email Hossein Hasan at hossein.h.hasan@gmail.com.
