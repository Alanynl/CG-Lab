# Taichi 贝塞尔与 B 样条曲线绘制实验
计算机图形学实验 - 基于 Taichi 的交互式 2D 参数曲线渲染

## 项目介绍
本项目使用 Taichi 实现交互式贝塞尔曲线与 B 样条曲线绘制，包含基础版、反走样优化版、曲线切换版。

## 文件说明
- **main.py** — 基础贝塞尔曲线绘制（De Casteljau 算法）
- **optional_1.py** — 高亮反走样贝塞尔曲线
- **optional_2.py** — 贝塞尔 / B 样条曲线一键切换

## 运行方式
```bash
uv run -m src.Work2.main
uv run -m src.Work2.optional_1
uv run -m src.Work2.optional_2
