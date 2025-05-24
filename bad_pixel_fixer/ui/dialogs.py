import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import threading
import time

from ..i18n import get_string as _

class PreviewDialog:
    def __init__(self, parent, temp_file, original_file, fixer, last_save_dir, load_callback=None):
        self.parent = parent
        self.temp_file = temp_file
        self.original_file = original_file
        self.fixer = fixer
        self.last_save_dir = last_save_dir
        self.load_callback = load_callback
        self.result = None
        
        self.create_dialog()
    
    def create_dialog(self):
        preview = tk.Toplevel(self.parent)
        preview.title(_("preview_title"))
        preview.geometry("900x650")  # 增加高度以容纳质量选择
        
        # 加载原图和修复后的图像
        orig_img = cv2.imread(self.original_file)
        fixed_img = cv2.imread(self.temp_file)
        
        if orig_img is None or fixed_img is None:
            messagebox.showerror(_("error_title"), _("error_load_preview"))
            preview.destroy()
            try:
                os.remove(self.temp_file)
            except:
                pass
            return
            
        # 转换为RGB
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        fixed_rgb = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2RGB)
        
        # 创建PIL图像
        orig_pil = Image.fromarray(orig_rgb)
        fixed_pil = Image.fromarray(fixed_rgb)
        
        # 获取原始文件大小（MB）
        orig_size_mb = os.path.getsize(self.original_file) / (1024 * 1024)
        fixed_size_mb = os.path.getsize(self.temp_file) / (1024 * 1024)
        
        # 缩放图像以适应显示
        width, height = orig_pil.size
        max_size = 400
        if width > max_size or height > max_size:
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            orig_pil_display = orig_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            fixed_pil_display = fixed_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            orig_pil_display = orig_pil
            fixed_pil_display = fixed_pil
        
        # 创建Tkinter图像
        orig_tk = ImageTk.PhotoImage(orig_pil_display)
        fixed_tk = ImageTk.PhotoImage(fixed_pil_display)
        
        # 创建上半部分的对比显示框架
        compare_frame = tk.Frame(preview)
        compare_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        # 原图显示
        orig_frame = tk.LabelFrame(compare_frame, text=_("preview_original", size=orig_size_mb))
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        orig_label = tk.Label(orig_frame, image=orig_tk)
        orig_label.image = orig_tk  # 保持引用
        orig_label.pack(padx=5, pady=5)
        
        # 修复后图像显示
        fixed_frame = tk.LabelFrame(compare_frame, text=_("preview_fixed", size=fixed_size_mb))
        fixed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        fixed_label = tk.Label(fixed_frame, image=fixed_tk)
        fixed_label.image = fixed_tk  # 保持引用
        fixed_label.pack(padx=5, pady=5)
        
        # 下半部分显示放大区域
        zoom_frame = tk.Frame(preview)
        zoom_frame.pack(fill=tk.X, expand=True, padx=10, pady=5)
        
        # 放大显示区域
        zoom_info = tk.Label(zoom_frame, text=_("preview_zoom_tip"))
        zoom_info.pack()
        
        zoom_area_frame = tk.Frame(zoom_frame)
        zoom_area_frame.pack(fill=tk.X, expand=True)
        
        # 原图放大区域
        orig_zoom_frame = tk.LabelFrame(zoom_area_frame, text=_("preview_original_zoom", x1=0, y1=0, x2=0, y2=0))
        orig_zoom_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        orig_zoom_label = tk.Label(orig_zoom_frame)
        orig_zoom_label.pack(padx=5, pady=5)
        
        # 修复后放大区域
        fixed_zoom_frame = tk.LabelFrame(zoom_area_frame, text=_("preview_fixed_zoom", x1=0, y1=0, x2=0, y2=0))
        fixed_zoom_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        fixed_zoom_label = tk.Label(fixed_zoom_frame)
        fixed_zoom_label.pack(padx=5, pady=5)
        
        # 添加鼠标移动放大功能
        def show_zoom(event, label, image, zoom_label, prefix):
            if not hasattr(event, 'x') or not hasattr(event, 'y'):
                return
                
            # 获取原始图像中的对应位置
            img_width, img_height = image.size
            label_width = label.winfo_width()
            label_height = label.winfo_height()
            
            # 计算比例
            scale_x = img_width / label_width if label_width > 0 else 1
            scale_y = img_height / label_height if label_height > 0 else 1
            
            # 获取鼠标在原始图像中的位置
            img_x = int(event.x * scale_x)
            img_y = int(event.y * scale_y)
            
            # 提取放大区域
            zoom_size = 100  # 提取的区域大小
            crop_x1 = max(0, img_x - zoom_size//2)
            crop_y1 = max(0, img_y - zoom_size//2)
            crop_x2 = min(img_width, crop_x1 + zoom_size)
            crop_y2 = min(img_height, crop_y1 + zoom_size)
            
            # 确保区域大小固定
            if crop_x2 - crop_x1 < zoom_size:
                crop_x1 = max(0, crop_x2 - zoom_size)
            if crop_y2 - crop_y1 < zoom_size:
                crop_y1 = max(0, crop_y2 - zoom_size)
            
            # 提取区域
            try:
                crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # 放大
                zoom_display = crop.resize((200, 200), Image.Resampling.LANCZOS)
                
                # 更新放大视图
                zoom_photo = ImageTk.PhotoImage(zoom_display)
                zoom_label.config(image=zoom_photo)
                zoom_label.image = zoom_photo  # 保持引用
                
                # 更新坐标信息
                if prefix == "preview_original_zoom":
                    zoom_label.master.config(text=_("preview_original_zoom", x1=crop_x1, y1=crop_y1, x2=crop_x2, y2=crop_y2))
                else:
                    zoom_label.master.config(text=_("preview_fixed_zoom", x1=crop_x1, y1=crop_y1, x2=crop_x2, y2=crop_y2))
            except Exception as e:
                print(f"放大显示出错: {str(e)}")
        
        # 绑定鼠标移动事件
        orig_label.bind("<Motion>", lambda e: show_zoom(e, orig_label, orig_pil, orig_zoom_label, "preview_original_zoom"))
        fixed_label.bind("<Motion>", lambda e: show_zoom(e, fixed_label, fixed_pil, fixed_zoom_label, "preview_fixed_zoom"))
        
        # 添加质量选择区域
        quality_frame = tk.LabelFrame(preview, text=_("output_quality_settings"))
        quality_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 当前输出文件大小信息
        size_info = tk.Label(quality_frame, 
            text=_("size_info", orig=orig_size_mb, fixed=fixed_size_mb, percent=fixed_size_mb/orig_size_mb*100))
        size_info.pack(pady=5)
        
        # 质量选择框架
        quality_select_frame = tk.Frame(quality_frame)
        quality_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 质量选项
        quality_options = [
            (_("quality_low"), 25), 
            (_("quality_medium"), 50),
            (_("quality_high"), 75),
            (_("quality_very_high"), 95)
        ]
        
        quality_var = tk.IntVar(value=95)  # 默认值为极高质量
        
        quality_tip = tk.Label(quality_select_frame, text=_("select_quality"))
        quality_tip.pack(side=tk.LEFT, padx=5)
        
        for text, value in quality_options:
            rb = tk.Radiobutton(quality_select_frame, text=text, variable=quality_var, value=value)
            rb.pack(side=tk.LEFT, padx=10)
        
        # 按钮框架
        btn_frame = tk.Frame(preview)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_result():
            # 获取选择的质量
            quality = quality_var.get()
            
            # 确定文件扩展名
            file_ext = os.path.splitext(self.original_file)[1].lower()
            if file_ext not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
                file_ext = '.jpg'  # 默认为JPEG
                
            initial_dir = os.path.dirname(self.original_file) if self.original_file else self.last_save_dir
            initialfile = f"fixed_{os.path.basename(self.original_file)}"
            output_path = filedialog.asksaveasfilename(
                initialdir=initial_dir,
                defaultextension=file_ext,
                initialfile=initialfile,
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif"), (_("all_files"), "*.*")]
            )
            if output_path:
                try:
                    # 更新状态
                    parent_status_var = None
                    for child in self.parent.winfo_children():
                        if isinstance(child, tk.Frame) and hasattr(child, 'winfo_name') and child.winfo_name() == 'status_frame':
                            for status_child in child.winfo_children():
                                if isinstance(status_child, tk.Label) and hasattr(status_child, 'cget'):
                                    if 'textvariable' in status_child.keys():
                                        parent_status_var = status_child.cget('textvariable')
                                        break
                    
                    if parent_status_var:
                        parent_status_var.set(_("saving_result", quality=quality))
                    
                    # 直接修复并保存
                    if self.fixer.fix_image(self.original_file, output_path, quality):
                        # 设置结果以便调用者可以知道保存成功
                        self.result = output_path
                        
                        # 询问是否加载修复后的图像
                        if messagebox.askyesno(_("load_fixed_image"), _("load_fixed_prompt")):
                            if self.load_callback:
                                self.load_callback(output_path)
                                
                        preview.destroy()
                    else:
                        messagebox.showerror(_("error_title"), _("save_failed"))
                except Exception as e:
                    messagebox.showerror(_("error_title"), _("error_save_file", error=str(e)))
        
        # 添加按钮
        cancel_btn = tk.Button(btn_frame, text=_("cancel"), command=preview.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        save_btn = tk.Button(btn_frame, text=_("save"), command=save_result)
        save_btn.pack(side=tk.RIGHT, padx=5)
        
        # 清理临时文件
        def on_close():
            if os.path.exists(self.temp_file):
                try:
                    os.remove(self.temp_file)
                except:
                    pass
            preview.destroy()
            
        preview.protocol("WM_DELETE_WINDOW", on_close)


class BatchProcessDialog:
    def __init__(self, parent, fixer, current_image_path, last_open_dir, last_save_dir):
        self.parent = parent
        self.fixer = fixer
        self.current_image_path = current_image_path
        self.last_open_dir = last_open_dir
        self.last_save_dir = last_save_dir
        self.result = False
        self.input_dir = ""
        self.output_dir = ""
        
        self.create_dialog()
    
    def create_dialog(self):
        # 创建批处理对话框
        batch_dialog = tk.Toplevel(self.parent)
        batch_dialog.title(_("batch_title"))
        batch_dialog.geometry("700x550")  # 增加高度以容纳GPU并行度设置
        batch_dialog.resizable(True, True)
        batch_dialog.transient(self.parent)
        batch_dialog.grab_set()  # 模态对话框
        
        # 创建主框架 - 使用Grid布局而不是Pack
        main_frame = tk.Frame(batch_dialog, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 第一行：左侧输入输出文件夹，右侧文件类型和选项
        # 左侧设置
        left_frame = tk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # 输入文件夹
        input_frame = tk.LabelFrame(left_frame, text=_("input_folder"))
        input_frame.pack(fill=tk.X, pady=5)
        
        input_path_var = tk.StringVar()
        input_entry = tk.Entry(input_frame, textvariable=input_path_var)
        input_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # 输出文件夹
        output_frame = tk.LabelFrame(left_frame, text=_("output_folder"))
        output_frame.pack(fill=tk.X, pady=5)
        
        output_path_var = tk.StringVar()
        output_entry = tk.Entry(output_frame, textvariable=output_path_var)
        output_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # 右侧设置 - 文件类型和选项
        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # 文件类型选择框架
        filetype_frame = tk.LabelFrame(right_frame, text=_("file_types"))
        filetype_frame.pack(fill=tk.X, pady=5)
        
        # 创建常见文件类型选择
        ext_var = tk.StringVar(value=".jpg")
        ext_options = [(".jpg", "JPEG (*.jpg)"), 
                       (".jpeg", "JPEG (*.jpeg)"), 
                       (".png", "PNG (*.png)"), 
                       (".bmp", "BMP (*.bmp)"),
                       (".tif", "TIFF (*.tif)")]
        
        # 文件列表预览控件 (先定义供后续使用)
        preview_frame = tk.LabelFrame(main_frame, text=_("preview_files"))
        file_listbox = tk.Listbox(preview_frame, selectmode=tk.EXTENDED, height=10)
        preview_info = tk.Label(preview_frame, text=_("files_found", count=0, ext=""), anchor=tk.W)
        
        # 刷新文件列表函数
        def refresh_file_list():
            file_listbox.delete(0, tk.END)  # 清空列表
            
            input_dir = input_path_var.get().strip()
            if not input_dir or not os.path.isdir(input_dir):
                preview_info.config(text=_("select_valid_input"))
                return
            
            # 确定扩展名
            if custom_check_var.get():
                ext = custom_ext_var.get().strip()
                if not ext:
                    preview_info.config(text=_("enter_custom_ext"))
                    return
                if not ext.startswith("."):
                    ext = "." + ext
            else:
                ext = ext_var.get()
            
            try:
                files = [f for f in os.listdir(input_dir) if f.lower().endswith(ext.lower())]
                files.sort()  # 按名称排序
                
                for f in files:
                    file_listbox.insert(tk.END, f)
                
                if len(files) > 0:
                    preview_info.config(text=_("files_found", count=len(files), ext=ext))
                else:
                    preview_info.config(text=_("no_files_found", ext=ext))
                    
            except Exception as e:
                preview_info.config(text=_("read_files_error", error=str(e)))
        
        # 为文件类型创建单选按钮
        for ext, desc in ext_options:
            rb = tk.Radiobutton(filetype_frame, text=desc, variable=ext_var, value=ext, command=refresh_file_list)
            rb.pack(anchor=tk.W, padx=10, pady=2)
        
        # 自定义扩展名
        custom_frame = tk.Frame(filetype_frame)
        custom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        custom_check_var = tk.BooleanVar(value=False)
        
        def toggle_custom_ext():
            if custom_check_var.get():
                custom_entry.config(state=tk.NORMAL)
            else:
                custom_entry.config(state=tk.DISABLED)
            refresh_file_list()
        
        custom_check = tk.Checkbutton(custom_frame, text=_("file_type_custom"), variable=custom_check_var, command=toggle_custom_ext)
        custom_check.pack(side=tk.LEFT)
        
        custom_ext_var = tk.StringVar()
        custom_entry = tk.Entry(custom_frame, textvariable=custom_ext_var, width=8, state=tk.DISABLED)
        custom_entry.pack(side=tk.LEFT, padx=5)
        custom_entry.bind("<KeyRelease>", lambda e: refresh_file_list())
        
        # 处理选项
        options_frame = tk.LabelFrame(right_frame, text=_("options"))
        options_frame.pack(fill=tk.X, pady=5)
        
        skip_var = tk.BooleanVar(value=True)
        skip_check = tk.Checkbutton(options_frame, text=_("skip_existing"), variable=skip_var)
        skip_check.pack(anchor=tk.W, padx=10, pady=5)
        
        # 添加质量选择
        quality_frame = tk.Frame(options_frame)
        quality_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(quality_frame, text=_("output_quality")).pack(side=tk.LEFT)
        
        quality_var = tk.IntVar(value=95)
        quality_combo = ttk.Combobox(quality_frame, textvariable=quality_var, width=5, 
                                     values=[25, 50, 75, 95])
        quality_combo.pack(side=tk.LEFT, padx=5)
        
        # 添加GPU并行度选择（仅当GPU可用时）
        if self.fixer.has_cuda:
            # GPU并行度设置
            parallel_frame = tk.LabelFrame(right_frame, text=_("gpu_parallel_settings") if _("gpu_parallel_settings") != "gpu_parallel_settings" else "GPU并行设置")
            parallel_frame.pack(fill=tk.X, pady=5)
            
            # 并行级别控制
            parallel_inner_frame = tk.Frame(parallel_frame)
            parallel_inner_frame.pack(fill=tk.X, padx=10, pady=5)
            
            parallel_label = tk.Label(parallel_inner_frame, text=_("parallel_level") if _("parallel_level") != "parallel_level" else "并行级别")
            parallel_label.pack(side=tk.LEFT)
            
            # 根据GPU内存估算最大并行数
            max_parallel = 8  # 默认最大值
            
            # 创建并行级别下拉菜单
            parallel_var = tk.IntVar(value=self.fixer.parallel_level)
            parallel_values = list(range(1, max_parallel + 1))
            parallel_combo = ttk.Combobox(parallel_inner_frame, textvariable=parallel_var, width=5,
                                        values=parallel_values)
            parallel_combo.pack(side=tk.LEFT, padx=5)
            
            # 并行度说明
            parallel_info = tk.Label(parallel_frame, 
                                    text=_("parallel_info") if _("parallel_info") != "parallel_info" else 
                                    "并行级别表示同时处理的图像数量。更高的值可以提高处理速度，但会消耗更多GPU内存。",
                                    wraplength=200, justify=tk.LEFT)
            parallel_info.pack(fill=tk.X, padx=10, pady=5)
            
        # 文件夹浏览按钮函数
        def select_input_folder():
            # 从当前图像目录或上次打开目录开始
            initial_dir = os.path.dirname(self.current_image_path) if self.current_image_path else self.last_open_dir
            folder = filedialog.askdirectory(title=_("select_input_folder"), initialdir=initial_dir)
            if folder:
                input_path_var.set(folder)
                # 尝试自动设置输出文件夹为"输入文件夹/fixed"
                output_folder = os.path.join(folder, "fixed")
                output_path_var.set(output_folder)
                refresh_file_list()
        
        def select_output_folder():
            # 如果已有输入文件夹，从输入文件夹开始；否则从上次目录开始
            initial_dir = input_path_var.get().strip()
            if not initial_dir or not os.path.isdir(initial_dir):
                initial_dir = self.last_save_dir
            folder = filedialog.askdirectory(title=_("select_output_folder"), initialdir=initial_dir)
            if folder:
                output_path_var.set(folder)
        
        # 添加浏览按钮
        btn_input = tk.Button(input_frame, text=_("browse"), command=select_input_folder)
        btn_input.pack(side=tk.RIGHT, padx=5, pady=5)
        
        btn_output = tk.Button(output_frame, text=_("browse"), command=select_output_folder)
        btn_output.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 第二行：预览文件区域
        preview_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=10)
        
        # 文件列表与滚动条
        scrollbar = tk.Scrollbar(preview_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        file_listbox.config(yscrollcommand=scrollbar.set)
        file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=file_listbox.yview)
        
        # 预览信息标签
        preview_info.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # 第三行：底部按钮和状态区域
        bottom_frame = tk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # 状态和错误消息
        status_frame = tk.Frame(bottom_frame)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        status_var = tk.StringVar()
        status_label = tk.Label(status_frame, textvariable=status_var, fg="blue")
        status_label.pack(side=tk.LEFT, padx=5)
        
        error_var = tk.StringVar()
        error_label = tk.Label(status_frame, textvariable=error_var, fg="red")
        error_label.pack(side=tk.LEFT, padx=5)
        
        # 按钮区域
        button_frame = tk.Frame(bottom_frame)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        def validate_and_start():
            # 清除之前的错误
            error_var.set("")
            status_var.set(_("preparing"))
            
            # 获取输入和输出文件夹
            input_dir = input_path_var.get().strip()
            output_dir = output_path_var.get().strip()
            
            # 检查文件夹
            if not input_dir:
                error_var.set(_("select_valid_input"))
                return
            
            if not output_dir:
                error_var.set(_("select_output_folder"))
                return
                
            if not os.path.isdir(input_dir):
                error_var.set(_("input_folder_not_exist"))
                return
                
            # 获取文件扩展名
            if custom_check_var.get():
                ext = custom_ext_var.get().strip()
                if not ext:
                    error_var.set(_("enter_custom_ext"))
                    return
                if not ext.startswith("."):
                    ext = "." + ext
            else:
                ext = ext_var.get()
            
            # 获取质量设置
            quality = quality_var.get()
            
            # 获取并行级别设置（如果有）
            if self.fixer.has_cuda and 'parallel_var' in locals():
                self.fixer.parallel_level = parallel_var.get()
            
            # 获取文件列表
            try:
                files = [f for f in os.listdir(input_dir) if f.lower().endswith(ext.lower())]
                files_count = len(files)
                
                if files_count == 0:
                    error_var.set(_("no_files_error", ext=ext))
                    return
                    
                # 确认处理
                skip_existing = skip_var.get()
                
                # 询问确认
                parallel_info = ""
                if self.fixer.has_cuda:
                    parallel_info = _("parallel_confirm_info", level=self.fixer.parallel_level) if _("parallel_confirm_info") != "parallel_confirm_info" else f"并行级别: {self.fixer.parallel_level}"
                
                confirmation_message = _("batch_confirmation", count=files_count, ext=ext, input=input_dir, output=output_dir, quality=quality)
                if parallel_info:
                    confirmation_message += f"\n{parallel_info}"
                
                if not messagebox.askyesno(_("confirm_title"), confirmation_message):
                    return
                
                # 记住目录以便下次使用
                self.input_dir = input_dir
                self.output_dir = output_dir
                self.result = True
                
                # 关闭设置对话框
                batch_dialog.destroy()
                
                # 启动批处理
                self._start_batch_process(input_dir, output_dir, ext, files, skip_existing, quality)
                
            except Exception as e:
                error_var.set(_("batch_error", error=str(e)))
        
        # 添加按钮 - 确保它们有足够大的尺寸
        cancel_btn = tk.Button(button_frame, text=_("cancel"), width=12, command=batch_dialog.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=10, pady=5)
        
        start_btn = tk.Button(button_frame, text=_("start_process"), width=12, command=validate_and_start,
                             bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        start_btn.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # 配置网格布局权重，确保预览区域可以拉伸
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)  # 让预览区域可以拉伸
        
        # 初始化界面
        if self.current_image_path:
            # 如果当前有图像，尝试自动设置其所在文件夹为输入文件夹
            input_dir = os.path.dirname(self.current_image_path)
            if input_dir:
                input_path_var.set(input_dir)
                output_dir = os.path.join(input_dir, "fixed")
                output_path_var.set(output_dir)
                refresh_file_list()
        
        # 初始聚焦
        input_entry.focus_set()
        
        # 确保对话框正常显示和更新
        batch_dialog.update()
        
    def _start_batch_process(self, input_dir, output_dir, ext, files, skip_existing, quality=95):
        """实际开始批处理操作"""
        
        # 创建输出文件夹（如果不存在）
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror(_("error_title"), _("creating_folder_error", error=str(e)))
                return
        
        files_count = len(files)
        
        # 创建进度窗口
        progress_win = tk.Toplevel(self.parent)
        progress_win.title(_("batch_progress_title"))
        progress_win.geometry("400x220")  # 稍微增加高度以容纳并行信息
        progress_win.resizable(False, False)
        progress_win.transient(self.parent)
        
        # 添加并行信息
        if self.fixer.has_cuda and self.fixer.parallel_level > 1:
            parallel_info = _("gpu_parallel_info", level=self.fixer.parallel_level) if _("gpu_parallel_info") != "gpu_parallel_info" else f"GPU并行级别: {self.fixer.parallel_level}"
            tk.Label(progress_win, text=parallel_info).pack(pady=5)
        
        tk.Label(progress_win, text=_("processing_with_quality", quality=quality)).pack(pady=5)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_win, variable=progress_var, maximum=files_count, length=350)
        progress_bar.pack(padx=20, pady=10)
        
        status_label = tk.Label(progress_win, text=_("preparing"))
        status_label.pack(pady=5)
        
        count_label = tk.Label(progress_win, text=_("batch_progress", current=0, total=files_count, 
                                                  success=0, failed=0, skipped=0))
        count_label.pack(pady=5)
        
        cancel_button = tk.Button(progress_win, text=_("cancel"), state=tk.DISABLED)
        cancel_button.pack(pady=10)
        
        progress_win.update()
        
        # 禁用主界面
        self._set_parent_ui_state(False)
        
        # 用于取消处理的标志
        cancel_flag = [False]
        
        def stop_processing():
            cancel_flag[0] = True
            cancel_button.config(text=_("cancelling"), state=tk.DISABLED)
        
        cancel_button.config(command=stop_processing, state=tk.NORMAL)
        
        def process_thread():
            try:
                # 准备要处理的文件路径列表
                image_paths = [os.path.join(input_dir, filename) for filename in files]
                
                # 定义更新UI的回调函数
                def update_callback(current, filename, processed, failed, skipped):
                    progress_win.after(0, lambda: update_progress(current, filename, processed, failed, skipped))
                
                # 开始批处理
                processed, failed, skipped = self.fixer.batch_process_images(
                    image_paths, 
                    output_dir, 
                    quality, 
                    skip_existing, 
                    update_callback, 
                    cancel_flag
                )
                
                # 完成
                progress_win.after(0, lambda: finish_batch(processed, failed, skipped, cancel_flag[0]))
                
            except Exception as e:
                progress_win.after(0, lambda e=e: handle_batch_error(str(e)))
        
        def update_progress(current, filename, processed, failed, skipped):
            progress_var.set(current)
            count_label.config(text=_("batch_progress", current=current, total=files_count, 
                                    success=processed, failed=failed, skipped=skipped))
            status_label.config(text=_("processing", filename=filename))
        
        def finish_batch(processed, failed, skipped, was_cancelled):
            self._set_parent_ui_state(True)
            
            if was_cancelled:
                message = _("batch_cancelled")
            else:
                message = _("batch_complete", success=processed, failed=failed, skipped=skipped)
                
            status_label.config(text=message)
            cancel_button.config(text=_("close"), state=tk.NORMAL, command=progress_win.destroy)
            
            # 更新主窗口状态
            self._update_parent_status(message)
        
        def handle_batch_error(error_message):
            self._set_parent_ui_state(True)
            status_label.config(text=_("batch_error_message", error=error_message))
            cancel_button.config(text=_("close"), state=tk.NORMAL, command=progress_win.destroy)
            self._update_parent_status(_("batch_error"))
        
        # 处理窗口关闭
        def on_progress_close():
            if not cancel_flag[0]:  # 如果尚未取消，则标记取消
                cancel_flag[0] = True
                return  # 不立即关闭，等待线程取消
            
            progress_win.destroy()
            self._set_parent_ui_state(True)
            
        progress_win.protocol("WM_DELETE_WINDOW", on_progress_close)
        
        # 在后台线程中运行
        threading.Thread(target=process_thread, daemon=True).start()
    
    def _set_parent_ui_state(self, enabled):
        """启用或禁用父窗口上的UI元素"""
        # 查找父窗口上的BadPixelFixerGUI实例
        for attr_name in dir(self.parent):
            if hasattr(self.parent, attr_name) and attr_name.startswith('btn_'):
                btn = getattr(self.parent, attr_name)
                if isinstance(btn, tk.Button):
                    btn.config(state=tk.NORMAL if enabled else tk.DISABLED)
    
    def _update_parent_status(self, message):
        """更新父窗口上的状态栏"""
        if hasattr(self.parent, 'status_var'):
            self.parent.status_var.set(message)