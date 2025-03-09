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
        'numpy==1.24.4',
        'pandas==1.5.3',
        'scipy==1.10.1',
        'scikit-learn==1.2.2',
        'torchvision==0.19.0',
        'tqdm==4.67.0',
        'Pillow==10.1.0',
        'scanpy==1.10.3',
        'dgl==2.4.0'
    ]
)
