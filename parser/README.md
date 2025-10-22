pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cu118
conda install -y --override-channels -c "nvidia/label/cuda-11.8.0" \
  cuda-toolkit=11.8 cuda-nvcc=11.8

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
hash -r
which nvcc && nvcc --version   # 必须显示 release 11.8


pip install transformers
pip install huggingface_hub
pip install opencv-python
pip install numpy
pip install pillow
pip install open_clip_torch

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -e .

# 进入你已经准备好的环境
conda activate zys_parser

# 1) 固定用 conda 里的 CUDA 11.8，并避免误用系统 CUDA 头文件
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# 防止环境里设置过全局头文件路径干扰
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE

# 2) 确保用的就是你环境里的 pip/setuptools（可选但推荐）
python -m ensurepip -U || true
python -m pip install -U pip setuptools wheel ninja

# 3) 自检（应显示 cu118 / nvcc 11.8）
python - <<'PY'
import torch, os, subprocess, torch.utils.cpp_extension as ext
print("torch:", torch.__version__, "| torch.cuda:", torch.version.cuda)
print("CUDA_HOME (PyTorch):", ext.CUDA_HOME)
print("nvcc:", subprocess.getoutput("nvcc --version").splitlines()[-1])
PY

# 4) 安装 GroundingDINO —— 关键：不用 -e，禁用构建隔离
cd /mnt/data/disk2/zyu/ReasoningSAM/Parser/GroundingDINO
git clean -xfd || true
rm -rf build *.egg-info **/build **/*.so

PIP_NO_BUILD_ISOLATION=1 PIP_USE_PEP517=0 \
python -m pip install . --no-deps -v

# 5) 验证
python - <<'PY'
import groundingdino, torch
print("GroundingDINO OK ->", groundingdino.__file__)
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
PY
