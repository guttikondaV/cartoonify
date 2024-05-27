import streamlit as st

st.title("Conclusion")
st.image("https://miro.medium.com/v2/resize:fit:1400/1*go7sTFOGN2fJGgYrI3E-FA.png")
st.write(
    """
The following lessons were learned from this project:
1.	Most basic or free cloud offerings don’t work for real-world machine learning tasks. Be ready to spend money. When working on scale, it is better to optimize for money or use cloud-managed services.
2.	Don’t write the code for multi-device logic. Use tools like Lightning AI to modularize your code. Lightning AI takes care of moving between devices and other logic, allowing you to focus on the core logic.
3.	Keep monitoring your metrics. This helps to save money and preempt training thus reducing your experiment time. Neptune is a good option to monitor your experiments.
4.	Keep your image size and batch size in mind. If your image size is too large (e.g. 1024 x 1024), then your batch size will be limited to 1, which is very inefficient use of compute and storage.

To conclude, a complete model couldn’t be trained for the limitations of resources and other constraints. But the process of building this project proved to be invaluable and taught a lot of things that go into building a machine learning model in a professional and academic setting.
"""
)
