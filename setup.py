from setuptools import setup
import os

VERSION = "0.3.0"

def get_long_description():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    with open(readme_path, encoding="utf8") as fp:
        return fp.read()


setup(
    name="llmlib",
    version=VERSION,
    description="Portability lib for OpenAI, Hugging Face, and local model APIs",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Mark Watson",
    url="https://github.com/mark-watson/llmlib",
    project_urls={
        "Issues": "https://github.com/mark-watson/llmlib/issues",
        "Changelog": "https://github.com/mark-watson/llmlib/releases",
    },
    license="Apache License, Version 2.0",
    packages=["llmlib"],
    package_dir={"llmlib": "llmlib"},
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "openai": ["openai>=1.0"],
        "huggingface": ["transformers>=4.30", "torch>=2.0", "sentence-transformers>=2.2"],
        "chromadb": ["chromadb>=0.4"],
        "all": [
            "openai>=1.0",
            "transformers>=4.30",
            "torch>=2.0",
            "sentence-transformers>=2.2",
            "chromadb>=0.4",
        ],
        "dev": ["pytest>=7.0", "pytest-mock>=3.10"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
