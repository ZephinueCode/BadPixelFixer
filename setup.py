from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bad-pixel-fixer",
    version="1.3",
    author="Zephinue Wang",
    author_email="wangxc23@mails.tsinghua.edu.cn",
    description="A tool for detecting and fixing bad pixels in images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "torch>=1.7.0",
    ],
    entry_points={
        "console_scripts": [
            "bad-pixel-fixer=bad_pixel_fixer.main:main",
        ],
    },
)