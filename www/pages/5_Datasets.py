import streamlit as st

st.title("Datasets")

st.write(
    """
    - For this project, we will use the CelebA dataset and Cartoonset dataset to train the GAN.
    - The CelebA dataset is a large-scale face attributes dataset that contains over 2,000,000 images of celebrities. The dataset contains images of celebrities with annotations for various attributes such as hair
    colour, skin tone, and facial hair. We will use the images in the CelebA dataset to train the GAN to generate Memojis that look like the actual person.
    - The Cartoonset dataset is a dataset of cartoon images that contains over 1,000,000 images of cartoon characters. The dataset contains images of cartoon characters with annotations for various attributes such as hair colour, skin tone, and facial hair. We will use the images in the Cartoonset dataset to train the GAN to generate Memojis that look like cartoon characters.
    """
)

st.write("## Sample of datasets")

# Get image from static directory
st.markdown("![](app/static/samples.png)")
