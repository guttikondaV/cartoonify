import streamlit as st

st.title("Introduction")

st.image("https://cdsassets.apple.com/live/7WUAS350/images/ios/ios14-memoji-hero.jpg")

st.write(
    """
Memojis (or emojis) have become a significant part of our daily communication. They are a fun way to express our emotions and feelings. They also are a significant product of companies like Apple, Samsung and Google.

The challenge with creating Memojis is that they are unique to each person. This makes it difficult to create a general model that can generate Memojis for everyone. However, we can create a model that can generate Memojis that look like the actual person. They can resemble the person in terms of their hair colour, skin tone etc.

For this deep learning project, we will use a Generative Adversarial Network (GAN) to generate Memojis. We will use a dataset of images of people's faces to train the GAN. The GAN will learn the features of the faces and generate Memojis that look like the faces in the dataset.
         """
)

st.write("## What is a GAN?")
st.image(
    "https://cdn.clickworker.com/wp-content/uploads/2022/11/Generative-Adversarial-Networks-Architecture-scaled.jpg"
)
st.write(
    """A Generative Adversarial Network (GAN) is a type of deep learning model that is used to generate new data samples. It consists of two neural networks - a generator and a discriminator. The generator generates new data samples, while the discriminator tries to distinguish between real data samples and generated data samples. The two networks are trained together in a competitive setting, where the generator tries to fool the discriminator, and the discriminator tries to detect the generated data samples."""
)

st.write("## Inspiration")
st.image(
    "https://miro.medium.com/v2/resize:fit:916/format:webp/1*BXpjDD0O9YsTtVGvX62EeA.png"
)
st.write(
    """
    We will be using the CycleGAN architecture to generate Memojis. CycleGAN is a type of GAN that is used for image-to-image translation. It can learn to translate images from one domain to another without paired examples. This means that we can train the GAN on images of human faces and generate Memojis that look like the faces.

The GAN model consists of two neural networks - a generator and a discriminator. The generator takes random noise as input and generates new data samples, while the discriminator takes data samples and tries to distinguish between real data samples and generated data samples.

- In our project, we will try to use the paper [CycleGAN](https://arxiv.org/abs/1703.10593) to generate Memojis. CycleGAN is a type of GAN that is used for image-to-image translation. It can learn to translate images from one domain to another without paired examples. This means that we can train the GAN on images of human faces and generate Memojis that look like the faces.

- There are 2 Generators and 2 Discriminators in CycleGAN. The generators are responsible for translating images from one domain to another, while the discriminators are responsible for distinguishing between real and generated images.
"""
)

st.image(
    "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*jE6KgUrOSUM2lkLRWGwO4g.png"
)
st.write(
    """- Although for our use case, we only need one generator and one discriminator, we will still use the architecture of CycleGAN to generate Memojis."""
)

st.write("""### Sample""")
st.image(
    "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*KTWjn-D8GijMnzX7IRpN6g.png"
)
