# app.py
import streamlit as st
from PIL import Image
from Trained_model import predict

# Streamlit app
def main():
    st.title("Melanoma Detection Dashboard")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform prediction when the 'Predict' button is clicked
        if st.button('Predict'):
            likelihood_score = predict(uploaded_file)
            st.success(f'Likelihood score: {likelihood_score:.2f}')

if __name__ == '__main__':
    main()
