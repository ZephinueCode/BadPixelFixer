import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import cv2
import os
import threading
import time
import torch

from ..core.fixer import BadPixelFixerPyTorch
from ..i18n import get_string as _, get_language, set_language
from .dialogs import PreviewDialog, BatchProcessDialog

class BadPixelFixerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(_("app_title"))
        self.root.geometry("1280x800")
        self.fixer = BadPixelFixerPyTorch()
        
        self.current_image = None
        self.current_image_path = None
        self.display_image = None
        self.canvas_img_id = None
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        self.tool_mode = "add"  # 默认工具模式：add, remove
        
        # 记录最后使用的目录
        self.last_open_dir = os.getcwd()
        self.last_save_dir = os.getcwd()
        
        self.setup_ui()
        
        # 显示CUDA状态
        if self.fixer.has_cuda:
            self.status_var.set(_("status_cuda_enabled", device=torch.cuda.get_device_name(0)))
        else:
            self.status_var.set(_("status_cpu_mode"))
    
    def setup_ui(self):
        # 创建主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧工具栏
        self.toolbar = tk.Frame(self.main_frame, width=60, bd=1, relief=tk.RAISED)
        self.toolbar.pack(side=tk.LEFT, fill=tk.Y)
        
        # 添加工具按钮
        self.btn_open = tk.Button(self.toolbar, text=_("open_image"), width=10, command=self.open_image)
        self.btn_open.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_save = tk.Button(self.toolbar, text=_("save_result"), width=10, command=self.save_result)
        self.btn_save.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(self.toolbar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)  # 分隔线
        
        tool_label = tk.Label(self.toolbar, text=_("edit_tools"))
        tool_label.pack(padx=5, pady=2)
        
        self.btn_add = tk.Button(self.toolbar, text=_("add_pixel"), width=10, command=lambda: self.set_tool_mode("add"))
        self.btn_add.pack(fill=tk.X, padx=5, pady=2)
        
        self.btn_remove = tk.Button(self.toolbar, text=_("remove_pixel"), width=10, command=lambda: self.set_tool_mode("remove"))
        self.btn_remove.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Separator(self.toolbar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)  # 分隔线
        
        process_label = tk.Label(self.toolbar, text=_("processing"))
        process_label.pack(padx=5, pady=2)
        
        self.btn_detect = tk.Button(self.toolbar, text=_("auto_detect"), width=10, command=self.detect_bad_pixels)
        self.btn_detect.pack(fill=tk.X, padx=5, pady=2)
        
        self.btn_apply = tk.Button(self.toolbar, text=_("apply_fix"), width=10, command=self.apply_fix)
        self.btn_apply.pack(fill=tk.X, padx=5, pady=2)
        
        self.btn_clear = tk.Button(self.toolbar, text=_("clear_all"), width=10, command=self.clear_all_pixels)
        self.btn_clear.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Separator(self.toolbar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)  # 分隔线
        
        # 半径控制
        radius_frame = tk.Frame(self.toolbar)
        radius_frame.pack(fill=tk.X, padx=5, pady=5)
        
        radius_label = tk.Label(radius_frame, text=_("bad_pixel_radius"))
        radius_label.pack(side=tk.LEFT)
        
        self.radius_var = tk.IntVar(value=1)
        self.radius_spinbox = ttk.Spinbox(radius_frame, from_=1, to=10, width=3,
                                  textvariable=self.radius_var, command=self.update_radius)
        self.radius_spinbox.pack(side=tk.RIGHT)
        
        # 阈值控制
        threshold_frame = tk.Frame(self.toolbar)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        threshold_label = tk.Label(threshold_frame, text=_("detection_threshold"))
        threshold_label.pack(side=tk.LEFT)
        
        self.threshold_var = tk.IntVar(value=30)
        self.threshold_spinbox = ttk.Spinbox(threshold_frame, from_=10, to=50, width=3,
                                    textvariable=self.threshold_var)
        self.threshold_spinbox.pack(side=tk.RIGHT)
        
        ttk.Separator(self.toolbar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)  # 分隔线
        
        # 批处理按钮
        self.btn_batch = tk.Button(self.toolbar, text=_("batch_process"), width=10, command=self.batch_process)
        self.btn_batch.pack(fill=tk.X, padx=5, pady=5)
        
        # 加载/保存配置
        self.btn_save_config = tk.Button(self.toolbar, text=_("save_config"), width=10, command=self.save_config)
        self.btn_save_config.pack(fill=tk.X, padx=5, pady=2)
        
        self.btn_load_config = tk.Button(self.toolbar, text=_("load_config"), width=10, command=self.load_config)
        self.btn_load_config.pack(fill=tk.X, padx=5, pady=2)

        # 语言选择
        ttk.Separator(self.toolbar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)  # 分隔线
        
        lang_frame = tk.Frame(self.toolbar)
        lang_frame.pack(fill=tk.X, padx=5, pady=5)
        
        lang_label = tk.Label(lang_frame, text=_("language"))
        lang_label.pack(anchor=tk.W)
        
        self.lang_var = tk.StringVar(value=get_language())
        
        # 创建一个下拉菜单用于选择语言
        lang_zh = tk.Radiobutton(lang_frame, text=_("lang_chinese"), variable=self.lang_var, 
                                value="zh", command=self.change_language)
        lang_zh.pack(anchor=tk.W)
        
        lang_en = tk.Radiobutton(lang_frame, text=_("lang_english"), variable=self.lang_var, 
                                value="en", command=self.change_language)
        lang_en.pack(anchor=tk.W)
        
        ttk.Separator(self.toolbar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)  # 分隔线
        
        # 信息标签
        device_info = torch.cuda.get_device_name(0) if self.fixer.has_cuda else _("status_cpu_mode")
        info_text = _("app_info", torch_version=torch.__version__, device_info=device_info)
        
        info_label = tk.Label(self.toolbar, text=info_text, justify=tk.LEFT, wraplength=120)
        info_label.pack(padx=5, pady=5)
        
        # 创建右侧主内容区
        self.content_frame = tk.Frame(self.main_frame)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 图像显示区域（带滚动条）
        self.canvas_frame = tk.Frame(self.content_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.v_scrollbar = tk.Scrollbar(self.canvas_frame)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#333333", 
                               xscrollcommand=self.h_scrollbar.set,
                               yscrollcommand=self.v_scrollbar.set)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<Button-3>", self.canvas_right_click)  # 右键删除
        self.canvas.bind("<B2-Motion>", self.canvas_pan)
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)
        self.canvas.bind("<MouseWheel>", self.canvas_zoom)  # Windows
        self.canvas.bind("<Button-4>", lambda e: self.canvas_zoom(e, 1))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self.canvas_zoom(e, -1))  # Linux
        
        # 状态栏
        self.status_frame = tk.Frame(self.root, height=20, relief=tk.SUNKEN, bd=1)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.pixel_info_var = tk.StringVar()
        self.pixel_info_var.set(_("pixel_coord", x="---", y="---"))
        pixel_info_label = tk.Label(self.status_frame, textvariable=self.pixel_info_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        pixel_info_label.pack(side=tk.LEFT, fill=tk.Y)
        
        self.status_var = tk.StringVar()
        self.status_var.set(_("status_ready"))
        status_label = tk.Label(self.status_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.tool_mode_var = tk.StringVar()
        self.tool_mode_var.set(_("tool_add"))
        tool_mode_label = tk.Label(self.status_frame, textvariable=self.tool_mode_var, bd=1, relief=tk.SUNKEN, anchor=tk.E)
        tool_mode_label.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 鼠标移动显示像素信息
        self.canvas.bind("<Motion>", self.update_pixel_info)
        
        # 快捷键
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_result())
        self.root.bind("<Control-d>", lambda e: self.detect_bad_pixels())
        self.root.bind("<Control-a>", lambda e: self.apply_fix())
        self.root.bind("<Delete>", lambda e: self.clear_all_pixels())
    
    def change_language(self):
        """切换界面语言"""
        new_lang = self.lang_var.get()
        if set_language(new_lang):
            messagebox.showinfo(_("language"), _("language_changed"))
            
    def set_tool_mode(self, mode):
        self.tool_mode = mode
        if mode == "add":
            self.tool_mode_var.set(_("tool_add"))
            self.btn_add.config(relief=tk.SUNKEN)
            self.btn_remove.config(relief=tk.RAISED)
        else:
            self.tool_mode_var.set(_("tool_remove"))
            self.btn_add.config(relief=tk.RAISED)
            self.btn_remove.config(relief=tk.SUNKEN)
    
    def update_radius(self):
        try:
            radius = int(self.radius_var.get())
            if radius < 1:
                radius = 1
            if radius > 10:
                radius = 10
            self.fixer.current_radius = radius
        except:
            self.fixer.current_radius = 1
    
    def update_pixel_info(self, event):
        if self.current_image is not None:
            # 转换坐标到原图
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
            
            if 0 <= img_x < self.current_image.shape[1] and 0 <= img_y < self.current_image.shape[0]:
                # 获取像素值
                try:
                    b, g, r = self.current_image[img_y, img_x]
                    self.pixel_info_var.set(_("pixel_info", x=img_x, y=img_y, r=r, g=g, b=b))
                except:
                    self.pixel_info_var.set(_("pixel_coord", x=img_x, y=img_y))
    
    def clear_all_pixels(self):
        """清除所有坏点标记"""
        if not self.fixer.bad_pixels:
            return
            
        if messagebox.askyesno(_("confirm_title"), _("confirm_clear_pixels")):
            self.fixer.bad_pixels = []
            self.draw_bad_pixels()
            self.status_var.set(_("all_pixels_cleared"))
    
    def open_image(self):
        # 从最后打开的目录开始
        initial_dir = os.path.dirname(self.current_image_path) if self.current_image_path else self.last_open_dir
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[(_("image_files"), "*.jpg *.jpeg *.png *.bmp *.tif"), (_("all_files"), "*.*")]
        )
        if file_path:
            self.last_open_dir = os.path.dirname(file_path)
            self.load_image(file_path)
    
    def load_image(self, file_path):
        try:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror(_("error_title"), _("error_load_image", path=file_path))
                return
                
            self.fixer.image_size = (self.current_image.shape[0], self.current_image.shape[1])
            self.zoom_factor = 1.0  # 重置缩放
            self.update_canvas()
            self.status_var.set(_("image_loaded", filename=os.path.basename(file_path)))
            
            # 更新窗口标题
            self.root.title(f"{_('app_title')} - {os.path.basename(file_path)}")
            
            # 更新最后使用的目录
            self.last_open_dir = os.path.dirname(file_path)
        except Exception as e:
            messagebox.showerror(_("error_title"), _("error_loading_image", error=str(e)))
    
    def update_canvas(self):
        if self.current_image is None:
            return
            
        try:
            # 转换OpenCV图像到PIL格式
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # 应用缩放
            current_width, current_height = pil_img.size
            new_width = int(current_width * self.zoom_factor)
            new_height = int(current_height * self.zoom_factor)
            
            if new_width > 0 and new_height > 0:  # 避免尺寸为0
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 更新显示图像
            self.display_image = ImageTk.PhotoImage(pil_img)
            
            # 调整画布大小
            self.canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            # 显示图像
            if self.canvas_img_id:
                self.canvas.delete(self.canvas_img_id)
            self.canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
            
            # 绘制坏点标记
            self.draw_bad_pixels()
        except Exception as e:
            messagebox.showerror(_("error_title"), _("error_update_canvas", error=str(e)))
    
    def draw_bad_pixels(self):
        if not self.current_image is None:
            try:
                # 清除之前的标记
                self.canvas.delete("bad_pixel")
                
                # 计算缩放比例
                scale_x = self.zoom_factor
                scale_y = self.zoom_factor
                
                for x, y, radius in self.fixer.bad_pixels:
                    # 计算显示尺寸上的坐标
                    display_x = int(x * scale_x)
                    display_y = int(y * scale_y)
                    display_radius = max(3, int(radius * scale_x))  # 最小3像素便于查看
                    
                    self.canvas.create_oval(
                        display_x - display_radius, 
                        display_y - display_radius,
                        display_x + display_radius, 
                        display_y + display_radius,
                        outline="red", width=2, tags="bad_pixel"
                    )
                
                # 更新状态栏
                self.status_var.set(_("total_bad_pixels", count=len(self.fixer.bad_pixels)))
            except Exception as e:
                print(f"绘制坏点时出错: {str(e)}")
    
    def canvas_click(self, event):
        if self.current_image is None:
            return
            
        # 获取画布坐标
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # 转换到图像坐标
        img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
        
        if 0 <= img_x < self.current_image.shape[1] and 0 <= img_y < self.current_image.shape[0]:
            if self.tool_mode == "add":
                self.fixer.add_manual_pixel(img_x, img_y)
                self.status_var.set(_("add_pixel_at", x=img_x, y=img_y))
            elif self.tool_mode == "remove":
                if self.fixer.remove_pixel(img_x, img_y):
                    self.status_var.set(_("remove_pixel_at", x=img_x, y=img_y))
                else:
                    self.status_var.set(_("no_pixel_to_remove", x=img_x, y=img_y))
            
            self.draw_bad_pixels()
    
    def canvas_right_click(self, event):
        """右键快速删除坏点"""
        if self.current_image is None:
            return
            
        # 获取画布坐标
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # 转换到图像坐标
        img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
        
        if 0 <= img_x < self.current_image.shape[1] and 0 <= img_y < self.current_image.shape[0]:
            if self.fixer.remove_pixel(img_x, img_y):
                self.status_var.set(_("remove_pixel_at", x=img_x, y=img_y))
            else:
                self.status_var.set(_("no_pixel_to_remove", x=img_x, y=img_y))
            
            self.draw_bad_pixels()
    
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """将画布坐标转换为图像坐标"""
        img_x = int(canvas_x / self.zoom_factor)
        img_y = int(canvas_y / self.zoom_factor)
        return img_x, img_y
    
    def start_pan(self, event):
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")  # 切换到移动光标
    
    def canvas_pan(self, event):
        if not self.is_panning:
            return
            
        dx = self.pan_start_x - event.x
        dy = self.pan_start_y - event.y
        
        self.canvas.xview_scroll(dx, "units")
        self.canvas.yview_scroll(dy, "units")
        
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def stop_pan(self, event):
        self.is_panning = False
        self.canvas.config(cursor="")  # 恢复默认光标
    
    def canvas_zoom(self, event, delta=None):
        if self.current_image is None:
            return
            
        try:
            # 处理不同平台的滚轮事件
            if delta is None:
                delta = event.delta
            
            # 确定缩放方向
            if delta > 0:
                factor = 1.1  # 放大
            else:
                factor = 0.9  # 缩小
            
            # 限制缩放范围
            new_zoom = self.zoom_factor * factor
            if 0.1 <= new_zoom <= 5.0:
                # 记住鼠标位置的图像坐标
                canvas_x = self.canvas.canvasx(event.x)
                canvas_y = self.canvas.canvasy(event.y)
                old_img_x, old_img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
                
                # 应用缩放
                self.zoom_factor = new_zoom
                self.update_canvas()
                
                # 计算新的画布坐标
                new_canvas_x = old_img_x * self.zoom_factor
                new_canvas_y = old_img_y * self.zoom_factor
                
                # 调整视图，使得鼠标下的点保持不变
                self.canvas.xview_moveto((new_canvas_x - event.x) / (self.current_image.shape[1] * self.zoom_factor))
                self.canvas.yview_moveto((new_canvas_y - event.y) / (self.current_image.shape[0] * self.zoom_factor))
        except Exception as e:
            print(f"缩放时出错: {str(e)}")
    
    def detect_bad_pixels(self):
        if self.current_image is None:
            messagebox.showerror(_("error_title"), _("error_no_image"))
            return
        
        try:
            # 从控件获取阈值
            threshold = self.threshold_var.get()
            
            self.status_var.set(_("detecting_bad_pixels"))
            self.root.update()
            
            # 禁用UI
            self.set_ui_state(False)
            
            def do_detect():
                try:
                    bad_pixels = self.fixer.detect_from_gray(self.current_image_path, threshold)
                    
                    # 在主线程中更新UI
                    self.root.after(0, lambda: self.after_detection(len(bad_pixels)))
                except Exception as e:
                    self.root.after(0, lambda: self.handle_error(_("error_detect_fail"), str(e)))
            
            # 在后台线程中运行检测
            threading.Thread(target=do_detect, daemon=True).start()
        except Exception as e:
            self.set_ui_state(True)
            messagebox.showerror(_("error_title"), _("error_detection", error=str(e)))
    
    def set_ui_state(self, enabled):
        """启用或禁用UI元素"""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_open.config(state=state)
        self.btn_save.config(state=state)
        self.btn_add.config(state=state)
        self.btn_remove.config(state=state)
        self.btn_detect.config(state=state)
        self.btn_apply.config(state=state)
        self.btn_clear.config(state=state)
        self.btn_batch.config(state=state)
        self.btn_save_config.config(state=state)
        self.btn_load_config.config(state=state)
    
    def after_detection(self, count):
        self.set_ui_state(True)
        self.status_var.set(_("detected_pixels", count=count))
        self.draw_bad_pixels()
    
    def handle_error(self, title, message):
        self.set_ui_state(True)
        messagebox.showerror(title, message)
        self.status_var.set(_("error_occurred"))
    
    def apply_fix(self):
        if self.current_image is None:
            messagebox.showerror(_("error_title"), _("error_no_image"))
            return
            
        if not self.fixer.bad_pixels:
            messagebox.showwarning(_("error_title"), _("error_no_bad_pixels"))
            return
            
        # 预览修复效果
        self.preview_fix()
    
    def preview_fix(self):
        try:
            # 创建临时文件
            temp_dir = os.path.dirname(self.current_image_path) if self.current_image_path else self.last_open_dir
            temp_file = os.path.join(temp_dir, f"_temp_preview_{int(time.time())}.jpg")
            
            # 状态更新
            self.status_var.set(_("fix_preview_generating"))
            self.root.update()
            
            # 禁用UI
            self.set_ui_state(False)
            
            # 使用默认高质量设置生成预览
            quality = 95
            
            def do_fix():
                try:
                    success = self.fixer.fix_image(self.current_image_path, temp_file, quality)
                    self.root.after(0, lambda: self.show_preview(temp_file, success))
                except Exception as e:
                    self.root.after(0, lambda: self.handle_error(_("error_title"), _("error_preview")))
                    try:
                        # 清理临时文件
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except:
                        pass
            
            # 在后台线程中执行修复
            threading.Thread(target=do_fix, daemon=True).start()
            
        except Exception as e:
            self.set_ui_state(True)
            messagebox.showerror(_("error_title"), _("error_show_preview", error=str(e)))
    
    def show_preview(self, temp_file, success):
        self.set_ui_state(True)
        
        if not success or not os.path.exists(temp_file):
            messagebox.showerror(_("error_title"), _("error_preview"))
            return
        
        try:
            # 创建预览对话框
            preview_dialog = PreviewDialog(
                self.root, 
                temp_file, 
                self.current_image_path, 
                self.fixer,
                self.last_save_dir,
                self.load_image
            )
            if preview_dialog.result:
                self.last_save_dir = os.path.dirname(preview_dialog.result)
                self.status_var.set(_("fixed_saved", filename=os.path.basename(preview_dialog.result)))
        except Exception as e:
            messagebox.showerror(_("error_title"), _("error_show_preview", error=str(e)))
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def save_result(self):
        self.apply_fix()
    
    def save_config(self):
        if not self.fixer.bad_pixels:
            messagebox.showerror(_("error_title"), _("error_no_pixels_to_save"))
            return
        
        # 从当前图像所在目录或上次保存目录开始
        initial_dir = os.path.dirname(self.current_image_path) if self.current_image_path else self.last_save_dir
        initialfile = f"badpixels_{os.path.basename(self.current_image_path).split('.')[0]}.json" if self.current_image_path else "badpixels.json"
        
        file_path = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            initialfile=initialfile,
            defaultextension=".json",
            filetypes=[(_("bad_pixel_config"), "*.json")]
        )
        if file_path:
            try:
                self.fixer.save_config(file_path)
                self.last_save_dir = os.path.dirname(file_path)
                self.status_var.set(_("config_saved", filename=os.path.basename(file_path)))
            except Exception as e:
                messagebox.showerror(_("error_title"), _("error_save_config", error=str(e)))
    
    def load_config(self):
        # 从当前图像所在目录或上次打开目录开始
        initial_dir = os.path.dirname(self.current_image_path) if self.current_image_path else self.last_open_dir
        
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[(_("bad_pixel_config"), "*.json")]
        )
        if file_path:
            try:
                timestamp = self.fixer.load_config(file_path)
                self.draw_bad_pixels()
                self.last_open_dir = os.path.dirname(file_path)
                self.status_var.set(_("config_loaded", filename=os.path.basename(file_path), timestamp=timestamp))
            except Exception as e:
                messagebox.showerror(_("error_title"), _("error_load_config", error=str(e)))
    
    def batch_process(self):
        if not self.fixer.bad_pixels:
            messagebox.showerror(_("error_title"), _("error_no_bad_pixels"))
            return
        
        # 创建批处理对话框
        dialog = BatchProcessDialog(
            self.root,
            self.fixer,
            self.current_image_path, 
            self.last_open_dir,
            self.last_save_dir
        )
        
        # 更新最后使用的目录
        if dialog.result:
            self.last_open_dir = dialog.input_dir
            self.last_save_dir = dialog.output_dir