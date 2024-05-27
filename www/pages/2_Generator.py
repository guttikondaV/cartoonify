import streamlit as st

st.title("Generator")

st.header("Generator and its Architecutre")

st.write(
    """
The generator of the GAN is responsible for generating new data samples. It takes random noise as input and generates new data samples that resemble the real data samples. The generator is trained to generate data samples that are indistinguishable from the real data samples. The generator is trained using backpropagation and gradient descent to minimize the difference between the generated data samples and the real data samples.
    """
)

st.write("## Architecture of the Generator")

st.image(
    "https://miro.medium.com/v2/resize:fit:1400/format:webp/0*xpzrkhjToausfj16.png"
)

st.code(
    """
        class Generator(nn.Module):
    \"\"\"
    Generator network for the GAN.
    \"\"\"

    def __init__(self: Self, in_channels: int = 3, out_channels: int = 3) -> None:
        \"\"\"
        Initializes the Generator network.

        Args:
            in_channels (int): The number of channels in the input image.
            out_channels (int): The number of channels in the output image.
        \"\"\"
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            conv_block.ConvBlock(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
            ),  # c7s1-64
            conv_block.ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3
            ),  # d128
            conv_block.ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3
            ),  # d256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            residual_block.ResidualBlock(in_channels=256, out_channels=256),  # R256
            conv_block.ConvBlock(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                use_transpose_convolution=True,
            ),  # u128
            conv_block.ConvBlock(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                use_transpose_convolution=True,
            ),  # u64
            conv_block.ConvBlock(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            ),  # c7s1-3
            nn.Sigmoid(),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Forward pass of the Generator network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        \"\"\"
        return self.layers(x)
""",
    language="python",
    line_numbers=True,
)
