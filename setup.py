from setuptools import setup
import os

VERSION = "0.02"

def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


setup(
    name="llmlib",
    description="Portability lib for OpenAI, Hugging Face, and FastChat local model APIs",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Mark Watson",
    url="https://github.com/mark-watson/llmlib",
    project_urls={
        "Issues": "https://github.com/mark-watson/llmlib/issues",
        "CI": "https://github.com/mark-watson/llmlib/actions",
        "Changelog": "https://github.com/mark-watson/llmlib/releases",
    },
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=["llmlib"],
   install_requires=["openai"],
    extras_require={
       
    },
    python_requires=">=3.7",
)
