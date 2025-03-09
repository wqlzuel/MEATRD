import setuptools

setuptools.setup(
    name="meatrd", # Replace with your own username
    version="0.0.1",
    author="",
    author_email="",
    description="An implementation of the MEATRD in Pytorch",
    long_description="",
    long_description_content_type="",
    url="https://github.com/wqlzuel/MEATRD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires = [
        'numpy'
        'pandas',
        'scipy',
        'scikit-learn',
        'torchvision',
        'tqdm',
        'Pillow',
        'scanpy',
        'dgl'
    ]
)
