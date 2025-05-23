import cv2
import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import time
from datetime import datetime

class BadPixelFixerPyTorch:
    def __init__(self):
        self.bad_pixels = []  # 格式: [(x, y, radius), ...]
        self.image_size = None  # 记录配置文件对应的图像尺寸
        self.current_radius = 1  # 默认半径
        
        # 检查CUDA支持
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.has_cuda else "cpu")
        
        if self.has_cuda:
            cuda_device = torch.cuda.get_device_name(0)
            print(f"找到CUDA设备: {cuda_device}")
            
            # 获取更多CUDA信息
            cuda_capability = torch.cuda.get_device_capability(0)
            cuda_version = torch.version.cuda
            print(f"CUDA版本: {cuda_version}, 计算能力: {cuda_capability}")
            
            # 设置更精确的浮点数计算（如果可用）
            torch.backends.cudnn.benchmark = True
        else:
            print("未找到CUDA设备，将使用CPU模式")
    
    def detect_from_gray(self, image_path, threshold=30, edge_width=5):
        """通过灰片检测异常像素，使用PyTorch加速，并排除边缘区域"""
        start_time = time.time()
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_size = gray.shape
        
        # 创建边缘掩码 - 排除图像边缘几个像素
        height, width = gray.shape
        edge_mask = np.ones_like(gray, dtype=bool)
        edge_mask[:edge_width, :] = False  # 上边缘
        edge_mask[-edge_width:, :] = False  # 下边缘
        edge_mask[:, :edge_width] = False  # 左边缘
        edge_mask[:, -edge_width:] = False  # 右边缘
        
        # 转换为PyTorch张量并移至设备
        if self.has_cuda:
            try:
                # 转换为张量
                gray_tensor = torch.from_numpy(gray).float().to(self.device)
                edge_mask_tensor = torch.from_numpy(edge_mask).to(self.device)
                
                # 创建高斯滤波核
                kernel_size = 5
                sigma = 1.5
                # 创建1D高斯核
                gauss_1d = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2+1)**2 / (2*sigma**2))
                gauss_1d = gauss_1d / gauss_1d.sum()  # 归一化
                
                # 扩展为2D核
                gauss_2d = torch.outer(gauss_1d, gauss_1d).to(self.device)
                gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size)  # 形状为[1, 1, k, k]
                
                # 应用高斯滤波
                gray_tensor = gray_tensor.view(1, 1, gray.shape[0], gray.shape[1])  # 添加批次和通道维度
                blurred_tensor = F.conv2d(gray_tensor, gauss_2d, padding=kernel_size//2)
                
                # 计算差异
                diff_tensor = torch.abs(gray_tensor - blurred_tensor)
                
                # 标准差计算 (用于区分纹理)
                mean_tensor = F.avg_pool2d(F.pad(gray_tensor, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
                sq_diff = (gray_tensor - mean_tensor) ** 2
                var_tensor = F.avg_pool2d(F.pad(sq_diff, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
                std_tensor = torch.sqrt(var_tensor + 1e-6)
                
                # 组合条件
                # 1. 差异大于阈值
                # 2. 标准差小于阈值的1.5倍(平坦区域) 或 差异极大(超过阈值2倍)
                # 3. 不在边缘区域
                cond1 = diff_tensor > threshold
                cond2 = (std_tensor < threshold*1.5) | (diff_tensor > threshold*2)
                
                # 最终掩码 - 添加边缘掩码
                mask_tensor = cond1 & cond2 & edge_mask_tensor.view(1, 1, height, width)
                
                # 提取坐标
                coords = torch.nonzero(mask_tensor.squeeze())  # [N, 2] 形状，每行是 [y, x] 坐标
                
                # 转换为NumPy并处理
                coords_np = coords.cpu().numpy()
                
                # 处理坐标
                bad_pixels = []
                for y, x in coords_np:
                    # 获取差异值，用于确定半径
                    diff_value = diff_tensor[0, 0, y, x].item()
                    radius = max(1, min(3, int(diff_value / threshold)))
                    bad_pixels.append((int(x), int(y), radius))
                
                print(f"使用PyTorch GPU检测到 {len(bad_pixels)} 个坏点，耗时: {time.time() - start_time:.2f} 秒")
                self.bad_pixels = bad_pixels
                return bad_pixels
                
            except Exception as e:
                print(f"PyTorch GPU处理失败，回退到CPU: {str(e)}")
                # 回退到CPU处理
        
        # CPU处理（如果GPU不可用或失败）
        print("使用CPU进行坏点检测")
        # 使用更大的窗口评估像素差异
        window_size = 3
        bad_pixels = []
        
        for y in range(window_size, gray.shape[0]-window_size):
            for x in range(window_size, gray.shape[1]-window_size):
                # 跳过边缘区域
                if not edge_mask[y, x]:
                    continue
                    
                # 获取窗口中的所有像素
                window = gray[y-window_size:y+window_size+1, x-window_size:x+window_size+1]
                center = gray[y, x]
                
                # 计算窗口中心与窗口平均值的差异
                window_mean = np.mean(window)
                diff = abs(center - window_mean)
                
                # 同时计算窗口的标准差，用于判断是否是纹理区域
                window_std = np.std(window)
                
                # 只在平坦区域检测异常（低标准差），或极度异常的像素
                if (diff > threshold and window_std < threshold*1.5) or diff > threshold*2:
                    # 智能设置半径：差异越大，半径越大
                    radius = max(1, min(3, int(diff / threshold)))
                    bad_pixels.append((x, y, radius))
        
        print(f"使用CPU检测到 {len(bad_pixels)} 个坏点，耗时: {time.time() - start_time:.2f} 秒")
        self.bad_pixels = bad_pixels
        return bad_pixels
    
    def add_manual_pixel(self, x, y, radius=None):
        """手动添加坏点"""
        if radius is None:
            radius = self.current_radius
        self.bad_pixels.append((x, y, radius))
    
    def remove_pixel(self, x, y, tolerance=5):
        """移除点击位置附近的坏点"""
        for i, (px, py, _) in enumerate(self.bad_pixels):
            if (px - x)**2 + (py - y)**2 <= tolerance**2:
                self.bad_pixels.pop(i)
                return True
        return False
    
    def fix_image(self, image_path, output_path, quality=95):
        """修复图像中的激光斑点，保持原始颜色平衡，自定义输出质量"""
        start_time = time.time()
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        # 检查图像尺寸
        if self.image_size and (img.shape[0] != self.image_size[0] or 
                            img.shape[1] != self.image_size[1]):
            print("警告：图像尺寸与配置文件不匹配！")
            return False
        
        # 保存原始图像的颜色统计信息（每个通道的均值）
        channel_means_original = [np.mean(img[:,:,i]) for i in range(3)]
        
        # 创建精确的坏点掩码（不过度扩展）
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for x, y, radius in self.bad_pixels:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:  # 边界检查
                # 使用原始半径，避免过度扩展
                cv2.circle(mask, (x, y), radius + 1, 255, -1)  # 只略微扩大1像素
        
        # 为过渡区域创建单独的掩码
        transition_mask = np.zeros_like(mask)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        transition_mask = dilated - mask  # 只在边缘周围创建窄的过渡区
        
        # 使用INPAINT_NS方法修复（保持边缘结构更好）
        inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        
        # 使用PyTorch进行GPU加速的精细修复
        if self.has_cuda:
            try:
                # 转换为PyTorch张量
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device)
                mask_tensor = torch.from_numpy(mask).float().to(self.device) / 255.0
                transition_tensor = torch.from_numpy(transition_mask).float().to(self.device) / 255.0
                inpainted_tensor = torch.from_numpy(inpainted).permute(2, 0, 1).float().to(self.device)
                
                # 为掩码区域创建平滑的修复
                padding = 2
                smoothed = F.avg_pool2d(
                    F.pad(inpainted_tensor.unsqueeze(0), (padding, padding, padding, padding), mode='reflect'),
                    5, stride=1
                ).squeeze(0)
                
                # 创建3通道掩码
                mask_3d = mask_tensor.unsqueeze(0).expand(3, -1, -1)
                transition_3d = transition_tensor.unsqueeze(0).expand(3, -1, -1)
                
                # 应用修复，仅在掩码区域内
                # 1. 直接修复区域使用inpainted结果
                # 2. 过渡区域使用原图和修复结果的混合
                blend_ratio = 0.7  # 过渡区域中原图的占比
                
                result_tensor = img_tensor * (1 - mask_3d - transition_3d) + \
                                inpainted_tensor * mask_3d + \
                                (img_tensor * blend_ratio + inpainted_tensor * (1 - blend_ratio)) * transition_3d
                
                # 转换回NumPy数组
                result = result_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                
                # 颜色校正：保持原始颜色平衡
                # 计算修复后的颜色统计
                channel_means_result = [np.mean(result[:,:,i]) for i in range(3)]
                
                # 只对掩码区域应用颜色校正
                for i in range(3):
                    if abs(channel_means_result[i] - channel_means_original[i]) > 1.0:
                        # 计算校正因子
                        factor = channel_means_original[i] / max(channel_means_result[i], 1.0)
                        # 只对掩码区域应用校正
                        correction_mask = (mask + transition_mask) > 0
                        result[:,:,i] = np.where(
                            correction_mask,
                            np.clip(result[:,:,i] * factor, 0, 255).astype(np.uint8),
                            result[:,:,i]
                        )
                
                print(f"使用PyTorch GPU修复完成，耗时: {time.time() - start_time:.2f} 秒")
                
            except Exception as e:
                print(f"PyTorch GPU处理失败，回退到CPU: {str(e)}")
                result = inpainted
                
                # 颜色校正
                channel_means_result = [np.mean(result[:,:,i]) for i in range(3)]
                correction_mask = (mask + transition_mask) > 0
                
                for i in range(3):
                    if abs(channel_means_result[i] - channel_means_original[i]) > 1.0:
                        factor = channel_means_original[i] / max(channel_means_result[i], 1.0)
                        result[:,:,i] = np.where(
                            correction_mask,
                            np.clip(result[:,:,i] * factor, 0, 255).astype(np.uint8),
                            result[:,:,i]
                        )
        else:
            # CPU处理
            print("使用CPU进行修复")
            
            # 使用简单的融合修复
            result = inpainted.copy()
            
            # 平滑过渡区
            for i in range(3):  # RGB通道
                blur = cv2.GaussianBlur(result[:,:,i], (5, 5), 0)
                # 只在过渡区应用模糊
                result[:,:,i] = np.where(transition_mask > 0, blur, result[:,:,i])
            
            # 颜色校正
            channel_means_result = [np.mean(result[:,:,i]) for i in range(3)]
            correction_mask = (mask + transition_mask) > 0
            
            for i in range(3):
                if abs(channel_means_result[i] - channel_means_original[i]) > 1.0:
                    factor = channel_means_original[i] / max(channel_means_result[i], 1.0)
                    result[:,:,i] = np.where(
                        correction_mask,
                        np.clip(result[:,:,i] * factor, 0, 255).astype(np.uint8),
                        result[:,:,i]
                    )
            
            print(f"使用CPU修复完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 根据文件扩展名和质量设置保存图像
        file_ext = os.path.splitext(output_path)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg']:
            # JPEG格式，使用质量参数
            cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif file_ext == '.png':
            # PNG格式，使用压缩级别参数
            # PNG压缩级别：0-9，0表示无压缩，9表示最大压缩
            # 将质量转换为压缩级别：95->1, 75->3, 50->6, 25->9 (反向关系)
            compression_level = min(9, max(0, int(10 - quality / 10)))
            cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        else:
            # 其他格式或不支持质量设置的格式
            cv2.imwrite(output_path, result)
        
        return True
    
    def save_config(self, filename):
        """保存坏点配置"""
        config = {
            "image_size": self.image_size,
            "bad_pixels": self.bad_pixels,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cuda_available": self.has_cuda,
            "device": str(self.device)
        }
        with open(filename, 'w') as f:
            json.dump(config, f)
    
    def load_config(self, filename):
        """加载坏点配置"""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        self.image_size = tuple(config.get("image_size")) if config.get("image_size") else None
        self.bad_pixels = config.get("bad_pixels", [])
        
        return config.get("timestamp", "未知")