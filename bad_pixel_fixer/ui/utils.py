import os
import tkinter as tk
from tkinter import simpledialog, messagebox

def ask_quality(parent, current_quality=95):
    """弹出对话框询问输出质量"""
    from ..i18n import get_string as _
    
    # 创建一个简单对话框
    quality = simpledialog.askinteger(
        _("quality_dialog_title"),
        _("quality_dialog_message"),
        parent=parent,
        initialvalue=current_quality,
        minvalue=1,
        maxvalue=100
    )
    
    return quality if quality is not None else current_quality

def get_file_size_info(file_path):
    """获取文件大小信息"""
    if not os.path.exists(file_path):
        return 0, "0 B"
    
    size_bytes = os.path.getsize(file_path)
    
    # 转换为适当的单位
    units = ["B", "KB", "MB", "GB"]
    size = size_bytes
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return size_bytes, f"{size:.2f} {units[unit_index]}"

def create_tooltip(widget, text):
    """为控件创建工具提示"""
    def enter(event):
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25
        
        # 创建工具提示窗口
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack()
        
        widget.tooltip = tooltip
    
    def leave(event):
        if hasattr(widget, "tooltip"):
            widget.tooltip.destroy()
            del widget.tooltip
    
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)