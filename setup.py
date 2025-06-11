
from setuptools import setup, find_packages

setup(
    name='adv_neural_crypto',
    version='0.1.0',
    description='Adversarial Neural Cryptography: training encryption models with adversarial objectives.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'pytorch-lightning>=2.0.0',
        'pyyaml',
        'captum'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Security :: Cryptography',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
