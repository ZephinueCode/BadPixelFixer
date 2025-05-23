import os
import json
from importlib import import_module

# 从配置文件中加载语言设置，默认为中文
_config_path = os.path.join(os.path.expanduser("~"), ".bad_pixel_fixer_config.json")

# 默认语言
_current_language = "zh"

# 尝试加载配置
try:
    if os.path.exists(_config_path):
        with open(_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            _current_language = config.get("language", "zh")
except:
    pass  # 加载失败时使用默认值

# 加载语言模块
try:
    if _current_language == "en":
        from .strings_en import STRINGS
    else:
        from .strings_zh import STRINGS
except ImportError:
    # 作为回退方案
    from .strings_zh import STRINGS

def get_string(key, **kwargs):
    """获取指定键的字符串，支持格式化参数"""
    if key in STRINGS:
        try:
            return STRINGS[key].format(**kwargs)
        except:
            return STRINGS[key]
    return key

def get_language():
    """获取当前语言"""
    return _current_language

def set_language(language):
    """设置界面语言并保存到配置文件"""
    global _current_language
    _current_language = language
    
    # 保存设置到配置文件
    config = {}
    if os.path.exists(_config_path):
        try:
            with open(_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except:
            pass
    
    config["language"] = language
    
    try:
        with open(_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except:
        pass
    
    # 无法在运行时热切换语言模块，需要重启程序
    return True

def reload_strings():
    """重新加载字符串资源"""
    global STRINGS
    
    # 动态导入对应语言模块
    lang_module = import_module(f".strings_{_current_language}", package="bad_pixel_fixer.i18n")
    STRINGS = lang_module.STRINGS