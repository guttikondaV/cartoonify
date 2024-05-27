import streamlit as st

st.title("Loss Functions")

st.write(
    "For this project, there are two proposed loss functions: **the adversarial loss** and the **cycle-consistency loss**.\n Each loss serves a specific purpose. Let's look at each of them."
)

st.header("Adversarial Loss")
st.image(
    "https://miro.medium.com/v2/resize:fit:830/format:webp/1*ZbL4gNWevk3MAMsdD8hb4Q.png"
)
st.write(
    """
The adversarial loss is used to train the generator and the discriminator in the GAN. The generator is trained to generate data samples that are indistinguishable from the real data samples, while the discriminator is trained to distinguish between real data samples and generated data samples. The adversarial loss is used to train the discriminator to correctly classify the data samples as real or generated and to train the generator to generate data samples that fool the discriminator.
    """
)

st.write("## Implementation")
st.code(
    """
class AdversarialLoss(nn.Module):
    \"\"\"
    Adversarial loss for the GAN.

    Used for both forward and backward models separately.
    \"\"\"

    def __init__(self: Self) -> None:
        super().__init__()

    def forward(self: Self, disc_pred: torch.Tensor, disc_actual: torch.Tensor):
        \"\"\"
        Calculates the adversarial loss for the GAN.

        Args:
            disc_pred (torch.Tensor): The discriminator prediction.
            disc_actual (torch.Tensor): The actual discriminator prediction.

        Returns:
            torch.Tensor: The adversarial loss.
        \"\"\"
        first_term = torch.square(disc_pred - 1)
        second_term = torch.square(disc_actual - 1)
        third_term = torch.square(disc_pred)

        return first_term + second_term + third_term
""",
    language="python",
    line_numbers=True,
)

st.write(
    "Instead of the log loss, I have used the square loss for the adversarial loss. This stabilizes the training and is easy to compute gradients."
)

st.header("Cycle-Consistency Loss")
st.image(
    "https://miro.medium.com/v2/resize:fit:732/format:webp/1*MVcaP2rkO4X_0INtb9qHAg.png"
)
st.write(
    """
The cycle-consistency loss is used to ensure that the generator can translate data samples from one domain to another and back without losing information. The cycle-consistency loss is used to train the generator to generate data samples that can be translated back to the original domain without losing information. This loss helps to ensure that the generator learns to generate data samples that are meaningful and retain the important features of the data samples.
    """
)
st.write("## Implementation")
st.code(
    """
class CyclicLoss(nn.Module):
    \"\"\"
    Cyclic loss for the GAN.

    Used for both forward and backward models together.
    \"\"\"

    def __init__(self: Self, lambda_multiplier: float = 10) -> None:
        super().__init__()
        self.lambda_multiplier = lambda_multiplier
        self.l1_loss = nn.L1Loss()

    def forward(
        self: Self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_g_x: torch.Tensor,
        g_f_y: torch.Tensor,
    ):
        \"\"\"
        Calculates the cyclic loss for the GAN.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The target tensor.
            f_g_x (torch.Tensor): The forward generator prediction.
            g_f_y (torch.Tensor): The backward generator prediction.

        Returns:
            torch.Tensor: The cyclic loss.
        \"\"\"
        first_term = self.l1_loss(f_g_x, x)
        second_term = self.l1_loss(g_f_y, y)

        return self.lambda_multiplier * (first_term + second_term)
""",
    language="python",
    line_numbers=True,
)
    
st.write("## Remarks")
st.write(
    """
- The adversarial loss is easy to compute and is used to train the generator and the discriminator in the GAN.
- The cycle-consistency loss is very important but it is very computationally huge. My model doesn't use this loss as I don't have the computational power to train the model with this loss.
- Both the losses are essential for the GAN to generate Memojis that look like the actual person. Removing either of the loss results in poor performance of the model.
- A trick is to serialize the losses: One train for specified epochs for the adversarial loss and then train for the cycle-consistency loss.
    """
)
