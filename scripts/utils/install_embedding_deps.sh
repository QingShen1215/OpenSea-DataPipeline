#!/bin/bash
# Install dependencies for BAYC image embeddings

echo "======================================"
echo "Installing Image Embedding Dependencies"
echo "======================================"

echo ""
echo "📦 Installing required packages..."
echo ""

# Install PyTorch (if not already installed)
echo "[1/4] Checking PyTorch..."
python3 -c "import torch" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ PyTorch already installed"
else
    echo "  Installing PyTorch..."
    pip install torch torchvision
fi

# Install CLIP (Recommended method)
echo ""
echo "[2/4] Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Install OpenAI (already installed)
echo ""
echo "[3/4] Checking OpenAI..."
python3 -c "import openai" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ OpenAI already installed"
else
    echo "  Installing OpenAI..."
    pip install openai
fi

# Install additional dependencies
echo ""
echo "[4/4] Installing additional packages..."
pip install pillow tqdm numpy

echo ""
echo "======================================"
echo "✓ Installation Complete!"
echo "======================================"
echo ""
echo "You can now run:"
echo "  python generate_image_embeddings.py --method clip"
echo ""
