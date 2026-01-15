# Retrieval

本目录已在 venv 中安装了 LangChain 相关依赖，并提供了一个最小示例以便快速运行与验证。

## 环境
- Python: 使用项目本地虚拟环境（.venv）
- 依赖列表：见 `requirements.txt`

## 安装依赖
在项目根目录运行：

```bash
# 使用本项目 venv 的 Python 运行 pip
./.venv/bin/python -m pip install -r requirements.txt
```

如需使用系统的 Python，请确保激活到相同的 venv：

```bash
# 激活 venv（macOS / Linux）
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行示例
最小示例位于 `langchain_example.py`：

```bash
./.venv/bin/python langchain_example.py
```

首次运行会自动从 Hugging Face 下载嵌入模型（`sentence-transformers/all-MiniLM-L6-v2`），根据网络情况可能耗时。

## 现有脚本
- `1.py`：与示例类似，也可使用 venv 运行。

```bash
./.venv/bin/python 1.py
```

## 常见问题
- 如果下载模型失败，检查网络或设置国内镜像。
- 若使用不同 Python 版本，请重新创建 venv 并安装依赖。
