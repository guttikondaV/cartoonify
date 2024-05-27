import os
import pathlib
import time
from typing import Self, Tuple

import PIL.Image as Image
import streamlit as st
import torch
import torch.nn as nn
import torchvision as tv_io


def load_checkpoint() -> dict | None:
    """
    Load the latest checkpoint from the checkpoints directory.

    Returns:
        dict | None: The checkpoint dictionary if a checkpoint is found else None.
    """
    CKPT_DIR = os.environ.get(
        "CKPT_DIR", pathlib.Path(__file__).parent.parent / "static/checkpoints"
    )

    if isinstance(CKPT_DIR, str):
        CKPT_DIR = pathlib.Path(CKPT_DIR)

    if not os.path.exists(CKPT_DIR):
        return None

    files_in_ckpt_dir = sorted(os.listdir(CKPT_DIR))

    if len(files_in_ckpt_dir) == 0:
        return None

    files_in_ckpt_dir = list(
        filter(lambda file_name: file_name.endswith(".tar"), files_in_ckpt_dir)
    )

    if len(files_in_ckpt_dir) == 0:
        return None

    latest_checkpoint_filename = max(
        files_in_ckpt_dir, key=lambda file_name: file_name.split("=")[-1].split(".")[0]
    )

    checkpoint_path = os.path.join(CKPT_DIR, latest_checkpoint_filename)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    return checkpoint


class ConvBlock(nn.Module):
    """
    This block is the basic building block of the generator and the
    discriminator. It consists of a convolutional layer, a batch
    normalization layer and a leaky ReLU activation function.

    This layer is the basis for all learning in the GAN.
    """

    def __init__(
        self: Self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        use_transpose_convolution: bool = False,
        use_leaky_relu: bool = False,
        leaky_relu_slope: float = 0.2,
    ):
        """
        Initializes the ConvBlock class.

        Parameters:

        - in_channels (int): The number of input channels.
        - out_channels (int): The number of output channels.
        - kernel_size (int or tuple): The size of the kernel. Must be an int or (int, int)
        - stride (int): The stride of the convolutional layer.
        - padding (int): The padding of the convolutional layer
        - use_transpose_convolution (bool): Whether to use transpose convolution or not.
        - use_leaky_relu (bool): Whether to use leaky ReLU or not.
        - leaky_relu_slope (float): The slope of the leaky ReLU activation function.
        """
        super(ConvBlock, self).__init__()

        conv_block_to_use = (
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="reflect",
                dtype=torch.float32,
            )
            if not use_transpose_convolution
            else nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode="zeros",
                dtype=torch.float32,
            )
        )

        self.layers = nn.Sequential(
            conv_block_to_use,
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope) if use_leaky_relu else nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self: Self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                # Initialize the weights and bias with a normal distribution
                # This initialization is recommended by the authors of the paper
                nn.init.normal_(
                    layer.weight,
                    mean=float(os.environ.get("INIT_MEAN", 0.0)),
                    std=float(os.environ.get("INIT_STD", 0.02)),
                )
                nn.init.normal_(
                    layer.bias,
                    mean=float(os.environ.get("INIT_MEAN", 0.0)),
                    std=float(os.environ.get("INIT_STD", 0.02)),
                )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBlock class.

        Parameters:

        - x (torch.Tensor): The input tensor.

        Returns:

        - torch.Tensor: The output tensor.
        """
        return self.layers(x)


class ResidualBlock(nn.Module):
    """
    This is the block used in the ResNet architecture. It consists of two
    convolutional layers. The input is added to the output of the second
    convolutional layer.

    This block is used to create the discriminator in the GAN.

    Parameters:

    - in_channels (int): The number of input channels.
    - out_channels (int): The number of output channels.
    """

    def __init__(self: Self, in_channels: int, out_channels: int):

        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dtype=torch.float32,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dtype=torch.float32,
            ),
        )

        self._init_weights()

    def _init_weights(self: Self):
        for layer in self.layers:
            # Initialize the weights and bias with a normal distribution
            # This initialization is recommended by the authors of the paper
            nn.init.normal_(
                layer.weight,
                mean=float(os.environ.get("INIT_MEAN", 0.0)),
                std=float(os.environ.get("INIT_STD", 0.02)),
            )
            nn.init.normal_(
                layer.bias,
                mean=float(os.environ.get("INIT_MEAN", 0.0)),
                std=float(os.environ.get("INIT_STD", 0.02)),
            )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock class.

        Parameters:

        - x (torch.Tensor): The input tensor.

        Returns:

        - torch.Tensor: The output tensor.
        """
        return self.layers(x)


class Generator(nn.Module):
    """
    Generator network for the GAN.
    """

    def __init__(self: Self, in_channels: int = 3, out_channels: int = 3) -> None:
        """
        Initializes the Generator network.

        Args:
            in_channels (int): The number of channels in the input image.
            out_channels (int): The number of channels in the output image.
        """
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
            ),  # c7s1-64
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3),  # d128
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3),  # d256
            ResidualBlock(in_channels=256, out_channels=256),  # R256
            ResidualBlock(in_channels=256, out_channels=256),  # R256
            ResidualBlock(in_channels=256, out_channels=256),  # R256
            ResidualBlock(in_channels=256, out_channels=256),  # R256
            ResidualBlock(in_channels=256, out_channels=256),  # R256
            ResidualBlock(in_channels=256, out_channels=256),  # R256
            ResidualBlock(in_channels=256, out_channels=256),  # R256
            ResidualBlock(in_channels=256, out_channels=256),  # R256
            ResidualBlock(in_channels=256, out_channels=256),  # R256
            ConvBlock(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                use_transpose_convolution=True,
            ),  # u128
            ConvBlock(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                use_transpose_convolution=True,
            ),  # u64
            ConvBlock(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            ),  # c7s1-3
            nn.Sigmoid(),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.layers(x)


st.title("Demo")

# Write in bold that only Adversarial Loss is used and the model is trained on 10,000 images each.
st.write(
    "**Please note that the model is trained only on Adversarial Loss and is trained on 10,000 images each.  The outputs may not resemble the distribution of the emoji dataset.**"
)

checkpoint = load_checkpoint()

generator = Generator()

if checkpoint is not None:
    generator.load_state_dict(checkpoint["generator_state_dict"])
else:
    st.write(
        "No checkpoint found. The generator will be initialized with random weights and outputs will be imperceptible."
    )

generator.eval()

# generator.load_state_dict(checkpoint["generator_state_dict"])


# Write a header or subheader to say "Upload an Image"
st.header("Upload an Image")

# 2 column layout: 1st column for uploading images, 2nd column for taking pictures
col1, col2 = st.columns(2)

output_image = None
generation_time = None

# 1st column: Upload image
with col1:
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = tv_io.transforms.ToTensor()(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            start_time = time.perf_counter_ns()
            output_image = generator(image)
            end_time = time.perf_counter_ns()

            generation_time = (end_time - start_time) / 1e9

        output_image = output_image.squeeze(0).permute(1, 2, 0).numpy()


# 2nd column: Take picture
with col2:
    captured_input_image = st.camera_input("Capture an image")

    if captured_input_image is not None:
        image = Image.open(captured_input_image)
        image = tv_io.transforms.ToTensor()(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            start_time = time.perf_counter_ns()
            output_image = generator(image)
            end_time = time.perf_counter_ns()

            generation_time = (end_time - start_time) / 1e9

        output_image = output_image.squeeze(0).permute(1, 2, 0).numpy()

st.header("Output Image")
if output_image is not None:
    st.write(f"Generation Time: {generation_time:.2f} seconds")
    st.image(output_image, caption="Output Image", use_column_width=True)
