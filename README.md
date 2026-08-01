# CoAtXNet: A Dual-Stream Hybrid Transformer Based on Relative Cross-Attention for End-to-End Camera Localization from RGB-D images

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Husseinhhameed/CoAtXnet)
<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research-orange.svg)
![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-purple.svg)

</p>
## Overview

CoAtXNet is a hybrid model that leverages the strengths of both Convolutional Neural Networks (CNNs) and Transformers to enhance vision-based camera localization. By integrating RGB and depth images through cross-attention mechanisms, CoAtXNet significantly improves feature representation and bidirectional information flow between modalities. This approach combines the local feature extraction capabilities of CNNs with the global context modeling strengths of Transformers, resulting in superior performance across various indoor scenes.

![CoAtXNet Architecture](https://github.com/Husseinhhameed/CoAtXnet/blob/main/Architecture.png)

## Model Architecture

CoAtXNet is a dual-stream architecture that processes RGB and depth images in parallel. The core of the model consists of alternating convolutional and transformer blocks that allow for information exchange between the two modalities at multiple stages.

-   **Dual Stream Input (`s0`, `s0x`):** The model begins with separate convolutional layers to process the RGB (`in_channels`) and depth (`aux_channels`) images independently, generating initial feature maps.

-   **Hybrid Stages (`s1` to `s4`):** The main body of the network consists of four stages. Each stage is a sequence of blocks that can be either convolutional (`XMBConv`) or transformer-based (`XTransformer`), as defined by the `block_types` parameter. These blocks are designed to handle two input streams and facilitate cross-modal interaction.

-   **`XMBConv` (Cross MBConv):** A modified MBConv (Mobile Inverted Bottleneck Convolution) block. It processes both streams in parallel. For downsampling, it uses a max-pooling and a projection layer in a residual connection.

-   **`XAttention` (Cross-Attention):** This is the core mechanism for fusing information. It's a relative self-attention module that has been modified to incorporate cross-attention. For a given stream (e.g., RGB), it calculates attention not only with itself but also with the other stream (depth). This allows features from one modality to influence the feature representation of the other. It also uses a learnable relative position bias table to encode spatial information.

-   **`XTransformer` (Cross-Transformer):** This block uses the `XAttention` module. It reshapes the 2D feature maps into sequences, applies layer normalization, and then passes them through the cross-attention mechanism. The output is then fed through a FeedForward network, and a residual connection is applied.

-   **Final Layers:** After the final hybrid stage (`s4`), the feature maps from both streams are average-pooled, concatenated, and passed through a series of fully-connected layers to produce the final 7-element output (3 for translation, 4 for quaternion rotation).

## Model Variants

The repository provides several pre-configured variants of CoAtXNet, differing in depth and width.

| Function | Num Blocks (L) | Channels (D) | Parameters |
| :--- | :--- | :--- | :--- |
| `coatxnet_0()` | `[2, 2, 3, 5, 2]` | `[64, 96, 192, 384, 768]` | 26M |
| `coatxnet_1()` | `[2, 2, 6, 14, 2]` | `[64, 96, 192, 384, 768]` | 47M |
| `coatxnet_2()` | `[2, 2, 6, 14, 2]` | `[128, 128, 256, 512, 1026]` | 86M |
| `coatxnet_3()` | `[2, 2, 6, 14, 2]` | `[192, 192, 384, 768, 1536]` | 165M |
| `coatxnet_4()` | `[2, 2, 12, 28, 2]` | `[192, 192, 384, 768, 1536]`| 306M |

## Usage

You can instantiate a model and pass dummy data through it as follows. This example uses `coatxnet_0`.

```python
import torch
from Model import coatxnet_0

# Create dummy input tensors for RGB and Depth images
# Batch size = 1, RGB channels = 3, Depth channels = 1, Image size = 256x256
img = torch.randn(1, 3, 256, 256)
aux = torch.randn(1, 1, 256, 256)

# Instantiate the model
net = coatxnet_0()

# Forward pass
output = net(img, aux)

# Print the output shape
# The output will be of shape [batch_size, 7] for position and orientation
print(output.shape) 
# Expected output: torch.Size([1, 7])
```

## Dataset

The model is designed to be trained and evaluated on RGB-D datasets for camera localization. The following are standard benchmarks for this task.

### 7Scenes Dataset

The 7Scenes dataset is a widely used benchmark for evaluating vision-based camera localization methods. It contains sequences of RGB-D images captured in seven different indoor scenes with a handheld Kinect RGB-D camera. Each scene presents various challenges such as motion blur, varying lighting conditions, and texture-less surfaces, making it ideal for testing the robustness of camera localization algorithms.

**Scenes**
- Chess
- Fire
- Heads
- Office
- Pumpkin
- Kitchen
- Stairs

**Download**
You can download the dataset from [https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

**Reference**
For more details, refer to the original paper: [Real-Time RGB-D Camera Relocalization](https://www.microsoft.com/en-us/research/publication/real-time-rgb-d-camera-relocalization/).

### 12Scenes Dataset

The 12Scenes dataset is another comprehensive benchmark for camera localization, designed for large-scale indoor environments. It consists of sequences of RGB-D images captured in 12 different scenes, providing a more diverse set of challenges than the 7Scenes dataset, including longer sequences and more complex geometry. This dataset is ideal for testing localization algorithms in more intricate environments.

**Scenes**
- Apt1 
- Apt2 
- Office1 
- Office2
...and more.

**Download**
You can download the 12Scenes dataset from [https://graphics.stanford.edu/projects/reloc/#data/](https://graphics.stanford.edu/projects/reloc/#data/).

**Reference**
For more details, refer to the original paper: [Scene Coordinate Regression for Camera Relocalization](https://graphics.stanford.edu/projects/reloc/).
