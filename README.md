StyleGAN2 Image Generation API
This project provides a FastAPI-based service that generates images using a pre-trained StyleGAN2 model. Users can interact with the API to generate images based on random seeds, truncation values, and noise modes. The service can optionally support class-conditional generation if the model is trained with class labels.

Additionally, a simple user interface (UI) is provided for easy interaction with the API, allowing users to visualize the generated images directly from their browser.

Features
Generate images using a pre-trained StyleGAN2 model.
Optionally specify random seeds for reproducible image generation.
Control the variety of generated images with the truncation psi parameter.
Select the noise mode ('const', 'random', 'none') to control noise behavior.
Optionally generate images with class labels for conditional generation.
View generated images directly in the browser via a simple UI.
Prerequisites
Before running the application, ensure you have the following installed:

Python 3.7+
pip (Python package manager)
You will also need to have a pre-trained StyleGAN2 model file (typically a .pkl file).

Installation
1. Clone or Download the Repository
bash
Copy code
git clone https://github.com/yourusername/stylegan2-fastapi.git
cd stylegan2-fastapi
2. Install Required Dependencies
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
Then install the required dependencies:

bash
Copy code
pip install -r requirements.txt
requirements.txt should contain:

Copy code
fastapi
torch
dnnlib
Pillow
numpy
uvicorn
3. Download or Specify Pretrained StyleGAN2 Model
Ensure you have a pre-trained StyleGAN2 model file. You can use an existing .pkl file, or download one of the official models (such as those from NVIDIA). Once you have the model, place it in the directory or provide its URL when making requests.

Running the Application
1. Start the FastAPI server
To start the server, run:

bash
Copy code
uvicorn app:app --reload
This will start the FastAPI application on http://localhost:8800.

2. Access the User Interface
Open your web browser and navigate to http://localhost:8800. The user interface will be available where you can input the necessary parameters for image generation.

3. API Endpoints
/generate (POST)
Generates images based on the input parameters.

Request Body (JSON):

json
Copy code
{
  "network_pkl": "path_or_url_to_network.pkl",
  "seeds": [1, 2, 3], 
  "truncation_psi": 1.0,
  "noise_mode": "const",
  "class_idx": 0
}
Parameters:

network_pkl: URL or local path to the pre-trained StyleGAN2 model.
seeds: List of random seeds for image generation (optional).
truncation_psi: Truncation value to control the variety of generated images (float between 0 and 2).
noise_mode: Type of noise to use ('const', 'random', 'none').
class_idx: (Optional) Class index for conditional image generation.
Response:

The generated image will be returned as a PNG image.

How It Works
User Interface: The frontend (HTML form) allows users to input the parameters for image generation and submit them via a POST request to the /generate endpoint.
Image Generation: The backend (FastAPI) loads the pre-trained StyleGAN2 model, generates the image based on the provided random seeds and other parameters, and returns the image as a PNG file.
Display: The frontend receives the image and displays it directly in the browser.
File Structure
bash
Copy code
/project_folder
    /static
        index.html        # User Interface for generating images
    app.py                # FastAPI backend script
    requirements.txt      # Python dependencies
Optional Enhancements
Multiple Image Generation: If you want to generate multiple images for a batch of seeds, you can modify the backend to return all images together (e.g., in a ZIP file).
GPU Support: Modify the backend to use CUDA if a GPU is available for faster image generation.
Troubleshooting
CORS Issues: If you’re running the frontend and backend separately (e.g., frontend on a different port), you may need to enable CORS support in FastAPI by adding from fastapi.middleware.cors import CORSMiddleware and configuring it:

python
Copy code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your security needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Model Loading Errors: Ensure that the path or URL to the pre-trained model is correct. If using a URL, make sure it is accessible and valid.

License
This project is licensed under the MIT License - see the LICENSE file for details."# ImageGenerationUsingStyleGAN2" 
"# ImageGeneration" 
