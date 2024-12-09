import os
import re
from typing import List, Optional
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
import numpy as np
import PIL.Image
import dnnlib
import legacy
import uvicorn

app = FastAPI()

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]

# Generate images using StyleGAN2
@app.post("/generate")
def generate_images(
    network_pkl: str,
    seeds: Optional[List[int]] = None,
    truncation_psi: float = 1,
    noise_mode: str = 'const',
    outdir: str = 'generated_images'
):
    """
    Generate images using pretrained StyleGAN2 model.
    
    Parameters:
    - network_pkl: URL or local path to the pretrained network pickle file.
    - seeds: List of random seeds for image generation.
    - truncation_psi: Truncation value to control image variety.
    - noise_mode: Type of noise ('const', 'random', 'none').
    - outdir: Directory to save generated images.
    
    Returns:
    - A generated image file.
    """
    # Load network from pickle
    print(f"Loading networks from {network_pkl}...")
    device = torch.device('cpu')  # Change to 'cuda' if you want to use GPU
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Generate images
    if seeds is None:
        seeds = [42]  # Default seed if none provided

    # For generating a single image, we'll use the first seed
    seed = seeds[0]
    print(f"Generating image for seed {seed}...")

    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    # Save generated image
    image_path = os.path.join(outdir, f"seed{seed:04d}.png")
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(image_path)

    return FileResponse(image_path, media_type="image/png", headers={"Content-Disposition": f"attachment; filename=seed{seed:04d}.png"})

# Uvicorn Configuration (run FastAPI app)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
