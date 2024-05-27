import streamlit as st

st.title("Welcome to the Cartoonify App!")

# Write an introduction of the project
st.write(
    """
    This is a Streamlit app that demonstrates how to use a Generative Adversarial Network (GAN) to generate Memojis. The app uses a GAN to generate Memojis that look like the actual person or a cartoon character. The app uses the CelebA dataset and the Cartoonset dataset to train the GAN. The app uses the PyTorch library to implement the GAN. The app uses the Streamlit library to create the web interface
    """
)

st.header("Table of Contents")

# Create links to the different pages
st.page_link("./Cartoonify.py", label="Home", icon="🏠")
st.page_link("./pages/1_Introduction.py", label="Introduction", icon="📚")
st.page_link("./pages/2_Generator.py", label="Generator", icon="🔨")
st.page_link("./pages/3_Discriminator.py", label="Discriminator", icon="🛡️")
st.page_link("./pages/4_Losses.py", label="Losses", icon="💔")
st.page_link("./pages/5_Datasets.py", label="Datasets", icon="📊")
st.page_link("./pages/6_Demo.py", label="Demo", icon="🎥")
st.page_link("./pages/7_Conclusion.py", label="Conclusion", icon="🔚")

st.write(
    """
    ## Links
    - [GitHub Repository](https://github.com/guttikondaV/cartoonify)"""
)

st.write("")
# Add made "Made with ❤️ footer"
st.write("Made with 🔥 and 💻 by Varun Guttikonda")
