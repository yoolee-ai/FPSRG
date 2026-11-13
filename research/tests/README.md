# CPU tests / CPU 测试

From the repository root / 在仓库根目录运行：

```bash
python -m unittest discover -s research/tests -v
```

Python 3.11+ and NumPy are required. Tests cover signed targets, digit mappings,
native-final parsing, thought/final weights, same-node coverage and upstream cache
keys. They do not load weights or evaluate GPU throughput.

需要 Python 3.11+ 和 NumPy。测试覆盖正负目标、数字映射、完整回答解析、思考／回答权重、
同节点覆盖约束与上游缓存键；不加载权重，不测 GPU 吞吐。
