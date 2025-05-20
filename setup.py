from setuptools import setup, find_packages

setup(
    name="instruct_sam",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pulp",
        "open-clip-torch",
        "transformers",
        "openai",
        "opencv-python",
        "matplotlib",
        "pycocotools",
        "lvis",
        "tabulate",
        "aiohttp",
        "scikit-learn",
        "sentencepiece",
        "accelerate",
        "qwen-vl-utils[decord]==0.0.8",
    ]
)
