from setuptools import setup, find_packages

setup(
    name="ir_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pyserini==0.36.0",
        "faiss-cpu",
        "torch",
        "transformers",
        "sentence-transformers",
        "lightgbm",
        "tqdm",
        "pandas",
        "numpy"
    ]
)