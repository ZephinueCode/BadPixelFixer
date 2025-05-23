import tkinter as tk
import sys
from .ui.main_window import BadPixelFixerGUI

def main():
    """程序主入口"""
    # 创建主窗口
    root = tk.Tk()
    app = BadPixelFixerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()