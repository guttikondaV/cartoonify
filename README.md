# Emoji Generation using GANs

## Introduction

This project is about generating emojis using Generative Adversarial Networks (GANs). The dataset used for training the GANs is the [Cartoonset Dataset](https://google.github.io/cartoonset) from Google and [Celeb A dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

The Cartoonset contains nearly 1 million images of emojis and the Celeb A dataset contains nearly 2 million images of celebrities. The Cartoonset dataset is used to train the GANs to generate emojis and the Celeb A dataset is used to train the GANs to generate human faces.

## How to run the code
```python
python main.py
```

## Requirements
- Python 3.6
- PyTorch
- GPU (at least 32GB RAM)
- 500 GB of disk space
- 16 GB of RAM

## GAN Architecture
The GAN architecture used in this project is a CycleGAN. The CycleGAN architecture consists of a generator and a discriminator. The generator takes an image of a human face and generates an emoji. The discriminator takes an image of an emoji and predicts whether it is a real emoji or a fake emoji.

## Results
TBD

## References
- [CycleGAN](https://arxiv.org/abs/1703.10593)
- [XGAN](https://arxiv.org/abs/1711.05139)
- [EmojiGAN](https://arxiv.org/abs/1801.03818)
- [Emoji2Emoji](https://arxiv.org/abs/1801.03818)