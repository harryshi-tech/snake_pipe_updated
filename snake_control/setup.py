from setuptools import setup, find_packages

setup(
    name="snake_control",
    version="0.0.1",
    description="Controllers for snake_pipe (ROS-agnostic, ROS-ready).",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pyyaml",
        # Note: snake_bullet is a *local* sibling package in this repo. We purposely do
        # not declare it as a pip dependency here, otherwise `pip install -e snake_control`
        # will try to fetch a non-existent PyPI package. Use PYTHONPATH (recommended for
        # this repo), or `pip install -e snake_bullet` if you add packaging for it.
    ],
)
