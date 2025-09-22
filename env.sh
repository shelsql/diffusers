# Env for Wan2.2 Diffusers
conda create -n wan python=3.10
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -e ".[torch]"
pip install accelerate transformers safetensors
pip install xformers ftfy opencv-python

# Flash Attention 3 Installation
cd ..
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
pip install ninja packaging
pip install .
cd ../diffusers