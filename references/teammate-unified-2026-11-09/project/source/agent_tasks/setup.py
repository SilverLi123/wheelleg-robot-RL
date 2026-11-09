from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "psutil",
    "numpy>=1.26.0",
    # Isaac Sim 4.5 uses Python 3.10 / torch 2.5; 5.1 uses Python 3.11 / torch 2.7.
    "torch>=2.5.0,<2.6; python_version < '3.11'",
    "torchvision>=0.20.0,<0.21; python_version < '3.11'",
    "torch>=2.7.0; python_version >= '3.11'",
    "torchvision>=0.22.0; python_version >= '3.11'",
    # basic logger
    "tensorboard",
    # "pybullet",
    # "rosbags",
    # "pinocchio=3.1.0",
    "pytorch-kinematics"
]

# Installation operation
setup(
    name="agent_tasks",
    packages=["agent_tasks"],
    author="zzr",
    maintainer="zzr",
    url="",
    version="0.1.0",
    description="SCUT Robot Lab, Agent Tasks",
    keywords="Wheelbipe, Reinforcement Learning, Agent Tasks",
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10,<3.13",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
