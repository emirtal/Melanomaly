import streamlit as st
from PIL import Image
from predict_resnet import predict_resnet
from predict_clip import predict_CLIP

# Initialize session state for disclaimer acceptance
if 'accepted_disclaimer' not in st.session_state:
    st.session_state.accepted_disclaimer = False

def show_disclaimer():
    st.markdown("""
    # Disclaimer

    Please read the following disclaimer carefully before using this application.

    **Disclaimer:**
    - The information provided by this application is for informational purposes only.
    - It is not intended as a substitute for professional medical advice, diagnosis, or treatment.
    - Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    - Never disregard professional medical advice or delay in seeking it because of something you have read on this application.
    - Data Privacy and Security: Your uploaded images are processed locally on your machine without being sent to any cloud services or third parties. 

    **By clicking 'Accept', you acknowledge that you have read and understand this disclaimer and that you agree to its terms.**
    """)
    if st.button("Accept"):
        st.session_state.accepted_disclaimer = True


def show_dashboard():
    st.title('Melanomaly: Melanoma Detection Dashboard')
    st.sidebar.write("# ABCDEs of Melanoma")
    st.sidebar.write("""
    **A - Asymmetry:** One half is unlike the other half.  
    
    **B - Border:** An irregular, scalloped or poorly defined border.  
    
    **C - Color:** Is varied from one area to another; has shades of tan, brown, or black; or is sometimes white, red, or blue.  
    
    **D - Diameter:** Melanomas are usually greater than 6mm (the size of a pencil eraser) when diagnosed, but they can be smaller.  
    
    **E - Evolving:** A mole or skin lesion that looks different from the rest or is changing in size, shape, or color.
    """)

    # Sidebar with educational content
    st.sidebar.title('Educational Resources')
    st.sidebar.write('Learn more about melanoma, its symptoms, and preventive measures:')
    st.sidebar.markdown(
        '[American Cancer Society - Melanoma](https://www.cancer.org/cancer/types/melanoma-skin-cancer.html)')

    uploaded_file = st.file_uploader("Upload an image of suspected Melanoma here:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        CLIP_malignant_score, CLIP_benign_score = predict_CLIP(uploaded_file)
        resnet_malignant_score, resnet_benign_score, resnet_heatmap_path = predict_resnet(uploaded_file)

        # Perform predictions
        if st.button('Predict using CLIP Model'):
            st.write(f'Likelihood benign: {CLIP_benign_score:.4f}')
            st.write(f'Likelihood malignant: {CLIP_malignant_score:.4f}')

        if st.button('Predict using ResNet Model'):
            st.write(f'Likelihood benign: {resnet_benign_score:.4f}')
            st.write(f'Likelihood malignant: {resnet_malignant_score:.4f}')


            # Load and display heatmap if toggled
        if st.button('Toggle ResNet Grad-CAM Heatmap'):
            print('clicked')
            # Display Grad-CAM heatmap
            st.image(resnet_heatmap_path, caption='Grad-CAM Heatmap', use_column_width=True)

            # Description of Grad-CAM heatmap
            st.markdown("""
            ### Understanding the Grad-CAM Heatmap:
            The Grad-CAM heatmap visually highlights regions of the image that contribute most to the model's prediction.
            It helps interpret which parts of the image are influencing the model's decision between benign and malignant classifications.

            #### Heatmap Legend:
            - **Reddish Areas**: Indicates regions that ResNet sees as malignant.
            - **Bluish Areas**: Indicates regions that ResNet sees as benign.
            """)
            st.markdown("""
            #### About the Colors:
            The color intensity in the heatmap corresponds to the importance of the highlighted regions in the ResNet's decision-making process.
            """)

# Show disclaimer if not accepted, otherwise show the dashboard
if not st.session_state.accepted_disclaimer:
    show_disclaimer()
else:
    show_dashboard()
