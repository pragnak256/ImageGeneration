import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Title of the application
st.title('StyleGAN2 Image Generator')

# Input fields for parameters
network_pkl = st.text_input('Network Pickle URL or Path', 'http://localhost:8800/network.pkl')
seeds = st.text_input('Seeds (comma separated, e.g., 1,2,3)', '1')
truncation_psi = st.number_input('Truncation Psi', min_value=0.0, max_value=2.0, value=1.0, step=0.01)
noise_mode = st.selectbox('Noise Mode', ['const', 'random', 'none'])
class_idx = st.number_input('Class Index (Optional)', min_value=0, step=1, value=None)

# Button to generate the image
generate_button = st.button('Generate Image')

# If the button is pressed
if generate_button:
    # Convert seeds from string to list of integers
    seeds_list = [int(s.strip()) for s in seeds.split(',')]

    # Prepare the request data
    request_data = {
        "network_pkl": network_pkl,
        "seeds": seeds_list,
        "truncation_psi": truncation_psi,
        "noise_mode": noise_mode,
        "class_idx": class_idx if class_idx != 0 else None  # Class index is optional
    }

    # Send the POST request to the FastAPI backend
    with st.spinner('Generating image...'):
        try:
            # Send the request to the FastAPI backend
            response = requests.post('http://localhost:8800/generate', json=request_data)

            if response.status_code == 200:
                # Get the image from the response
                img = Image.open(BytesIO(response.content))

                # Display the image
                st.image(img, caption='Generated Image', use_column_width=True)
            else:
                st.error('Error generating image: ' + response.text)

        except Exception as e:
            st.error(f"Error generating image: {e}")
