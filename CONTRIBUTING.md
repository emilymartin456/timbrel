# 贡献指南

感谢你关注 Timbrel！这是一个处于研究早期阶段的项目，接口仍可能变动，非常欢迎 issue、讨论与 Pull Request。

## 开发环境

推荐 Python 3.9–3.12。

```bash
python -m venv .venv && source .venv/bin/activate
# CPU 版 PyTorch（按需替换为 CUDA 版本）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
```

## 本地检查

提交前请确保以下命令全部通过（CI 也会运行同样的检查）：

```bash
ruff format --check .
ruff check .
mypy timbrel
pytest
```

单元测试均在 CPU 上运行、使用极小的随机张量，不依赖任何数据集或预训练权重，因此整套测试可以在数秒内跑完。

## 代码风格

- 使用 `ruff` 统一格式与静态检查；行宽 100。
- 公共函数与类需要有类型标注和简短 docstring；注释只解释「为什么」，不复述「是什么」。
- 张量维度约定写在 docstring 里（例如 `(B, T, n_mels)`），方便他人对齐形状。

## 提交与 PR

- 每个 commit 只做一件事，信息尽量清晰；可用 `feat:` / `fix:` / `docs:` 等前缀，但不强制。
- 涉及行为变化时，请同步更新文档与 `CHANGELOG.md`。
- PR 请关联对应 issue，并在描述中说明动机与验证方式。

## 行为准则

参与本项目即表示你同意遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
