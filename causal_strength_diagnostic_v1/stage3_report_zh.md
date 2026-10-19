import sys
import platform
import importlib.util

print("Python版本：", sys.version)
print("Python路径：", sys.executable)
print("操作系统：", platform.platform())
print("是否已安装PyTorch：", importlib.util.find_spec("torch") is not None)