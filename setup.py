from setuptools import find_packages, setup


setup(
    name="neural-network-draw",
    version="0.1.0",
    description="Draw basic feed-forward neural network architecture diagrams with matplotlib",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["matplotlib>=3.7"],
    python_requires=">=3.8",
)
