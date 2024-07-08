# app.py
import streamlit as st
from PIL import Image
from predict_resnet import predict_resnet
from predict_clip import predict_CLIP

# Streamlit app
def main():
    st.title("Melanoma Detection Dashboard")

    # Model selection dropdown
    model_choice = st.selectbox("Select Model", ["CLIP Model", "ResNet Model"])

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform prediction based on the selected model when the 'Predict' button is clicked
        if st.button('Predict'):
            if model_choice == "CLIP Model":
                likelihood_score = predict_CLIP(uploaded_file)
                st.success(f'Likelihood score: {likelihood_score:.2f}')
            elif model_choice == "ResNet Model":
                likelihood_score = predict_resnet(uploaded_file)
                st.success(f'Likelihood score: {likelihood_score:.2f}')

if __name__ == '__main__':
    main()

#
