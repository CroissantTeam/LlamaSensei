from setuptools import find_packages, setup

with open("requirements-dev.txt") as f:
    required = f.read().splitlines()

setup(
    name="llama_sensei",
    version="0.0.1",
    description="A LLM-powered personal online courses teaching assistant",
    author="Croissant Team",
    packages=find_packages("app"),
    package_dir={"": "app"},
    install_requires=required,
)
