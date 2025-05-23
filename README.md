# Bad Pixel Fixer

![Version](https://img.shields.io/badge/version-1.3-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-supported-brightgreen)

An efficient tool for detecting and repairing bad pixels (such as laser spots, hot pixels) in images, with CUDA GPU acceleration and batch processing support.

## Features

- 🚀 **CUDA Acceleration**: Utilizes PyTorch and CUDA for GPU acceleration, significantly improving processing speed
- 🔍 **Intelligent Detection**: Automatically detects abnormal pixels in images
- ✏️ **Manual Editing**: Supports manual addition or removal of bad pixel markers
- 🖼️ **High-Quality Repair**: Implements advanced algorithms for seamless repair while maintaining original image quality
- 📦 **Batch Processing**: Process multiple images at once
- 💾 **Save/Load Configuration**: Save bad pixel configurations and apply them to other images taken with the same camera
- 🌐 **Multi-language Support**: Supports both Chinese and English interfaces

## Use Cases

- Repair bad pixels/hot pixels on camera sensors
- Clear abnormal bright spots in astrophotography
- Remove image flaws caused by laser points
- Fix minor imperfections in images

## Installation

### Source Code Installation

1. Clone the repository

2. Create a virtual environment (optional)

3. `pip install -r requirements.txt`

4. `pip install -e .`

### CUDA Support

For optimal performance, it's recommended to install a CUDA-supported version of PyTorch. Please visit the PyTorch official website to select and install the appropriate CUDA version for your system.

## Usage

```bash
bad-pixel-fixer
# or
python -m bad_pixel_fixer.main
```

The subsequent usage is quite straightforward - just follow the prompts!

### Recommendation

Try taking a photo with the affected camera without a lens, using medium gray exposure (or a white or black frame, depending on your camera's condition) to maximize the visibility of camera bad pixels. You can apply automatic detection to these types of photos and save the configuration file to achieve excellent results.

I hope no one else will have their camera damaged by laser pointers.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Issues and pull requests are encouraged.

## Contact

Contact me via Email! [Email](wangxc23@mails.tsinghua.edu.cn)
