import os
import sys
from app import create_app

# ==========================================
# 【工程环境配置】将项目根目录强制加入环境变量
# 防止任何由于运行路径不同导致的 ModuleNotFoundError
# ==========================================
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 调用我们在 app/__init__.py 中写好的工厂函数创建 Flask 实例
app = create_app()

if __name__ == '__main__':
    print("========================================")
    print("🚀 金融量化分析与回测平台 启动中...")
    print("👉 请打开浏览器访问: http://127.0.0.1:5000")
    print("========================================")

    # 开启 debug=True，代码有任何修改服务器会自动重启，极大地提升开发效率
    app.run(host='127.0.0.1', port=5000, debug=True)