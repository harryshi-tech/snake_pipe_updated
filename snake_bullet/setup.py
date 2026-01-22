from setuptools import setup, find_packages

setup(
    name="snake_bullet",
    version="0.0.1",
    description="PyBullet simulation backend for snake_pipe (ROS-agnostic, ROS-ready).",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "pybullet",
        "pyyaml",
        "numpy",
    ],
)
