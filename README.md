# Physics of Motion, Geometry of Cohesion: A Silky Gaussian Head Avatar Framework
** | [Project Page](https://example.com/project_page)**
*High-fidelity dynamic head avatars free from motion/geometry artifacts with spatiotemporal coherence*
Given a monocular video sequence, our framework reconstructs physics-plausible digital avatars in minutes and renders at 300+ FPS (512×512 resolution on RTX 3090) with natural facial dynamics and temporal consistency.
## 📋 Table of Contents
- [Key Features](#-key-features)
- [Setup](#-setup)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Inference](#-inference)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
---
## 🌟 Key Features
1. **Physics-Guided Motion Propagation**
   - Second-order kinematics for mean prediction
   - Lie group transformations on SPD manifold for covariance evolution
   - Adaptive Hellinger-based physical consistency loss
2. **Hierarchical Optimal Transport Regularization**
   - Anatomical grouping via FLAME landmarks
   - Hyperbolic-adaptive regularization weights
   - Differential elasticity equilibrium between rigid/mobile regions
3. **Coarse-to-Fine Architecture**
   - FLAME-anchored Gaussian initialization
   - UV Splatter Image deformation refinement
   - Real-time differentiable rasterization
---
## ⚙️ Setup
### System Requirements
- NVIDIA GPU with ≥24GB VRAM (RTX 3090 tested)
- CUDA 11.7+
- Python 3.8+
### Installation
```bash
# Create conda environment
conda create -n silkygauss python=3.8 -y
conda activate silkygauss
# Install core dependencies
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
# Install PyTorch3D (required for FLAME operations)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```
---
## 📂 Data Preparation
### 1. Input Video Processing
Place your monocular video in `data/original_videos/<your_video>.mp4`, then run:
```bash
python scripts/preprocess.py \
  --video_path data/original_videos/<your_video>.mp4 \
  --output_dir data/processed/<subject_id>
```
### 2. Identity & FLAME Parameters
1. **MICA Installation**
   Follow [MICA](https://github.com/Zielon/MICA) to extract identity coefficients:
   ```bash
   git clone https://github.com/Zielon/MICA.git
   cp -r MICA/demo data/mica_assets
   ```
2. **Metrical Tracker Setup**
   Install [metrical-tracker](https://github.com/Zielon/metrical-tracker) for FLAME parameter extraction:
   ```bash
   cd metrical-tracker
   python process_video.py --input ../data/processed/<subject_id> --output ../data/flame_params
   ```
### 3. Segmentation Masks
Generate facial parsing maps using [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch):
```bash
python scripts/generate_segmaps.py --input_dir data/processed/<subject_id>/frames
```
---
## 🚀 Training
### Single-Subject Training
```bash
python train.py \
  --config configs/base.yaml \
  --data_dir data/processed/<subject_id> \
  --flame_dir data/flame_params \
  --output_dir logs/<experiment_name> \
  --physics_weight 0.1 \
  --ot_weight 1.0 \
  --landmark_groups flame
```
**Key Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--physics_weight` | Weight for physical consistency loss | 0.1 |
| `--ot_weight` | Weight for hierarchical OT regularization | 1.0 |
| `--landmark_groups` | Landmark grouping strategy (`flame`/`face-parsing`) | flame |
---
## 🔮 Inference
### Novel Expression Rendering
```bash
python render.py \
  --checkpoint logs/<experiment_name>/checkpoint.pth \
  --flame_params data/flame_params/<animation_sequence>.npy \
  --output_path renders/<output_video>.mp4 \
  --resolution 512
```
### Interactive Editing
```python
from core.editor import AvatarEditor
editor = AvatarEditor("logs/<experiment_name>/checkpoint.pth")
editor.interactive_edit(
    expression="surprise",
    jaw_openness=0.7,
    output_path="edits/surprise_frame.png"
)
```

## 📜 Citation
```bibtex
@article{deng2025physics,
  title={Physics of Motion, Geometry of Cohesion: A Silky Gaussian Head Avatar Framework},
  author={Deng, Junli and Shi, Ping and Li, Qipei and Guo, Jinyang},
  journal={IEEE Signal Processing Letters},
  year={2025},
  publisher={IEEE}
}
```
---
## 📄 License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
---
## 🙏 Acknowledgments
- FLAME model by [MPI-IS](https://flame.is.tue.mpg.de/)
- MICA implementation by [Zielon](https://github.com/Zielon/MICA)
- Optimal Transport solver adapted from [GeomLoss](https://www.kernel-operations.io/geomloss/)