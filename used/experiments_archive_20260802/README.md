# Pixal3D 实验归档（2026-08-02）

本目录只保存已经完成的实验、诊断、smoke、测试代码和历史备份；它们不再参与默认 baseline 推理。

目录结构：

- `code/`：实验入口脚本、比较脚本、实验渲染/评测脚本。
- `tests/`：实验专用测试与回归测试。
- `outputs/`：原 `outputs/` 下全部实验结果，目录名和文件内容保持不变。
- `backups/`：所有旧 `copy`、`backup`、`.bak`、`.before_*`、`old` 和 superseded 文件；其中 `backups/legacy/` 保留原相对路径，包括核心 pipeline 的历史备份，不参与当前 import。
- `artifacts/`：旧 `metric_summaries/` 和 `logs/` 实验汇总/日志。
- `manifest.txt`：归档前后文件清单。
- `baseline_hashes.sha256`：归档前记录的 baseline 入口和 sampler hash。

保留在项目原位、未归档的 baseline 代码：

`inference.py`、`app.py`、`train.py`、`pixal3d/` 核心包、`data_toolkit/`、配置和资源文件。

官方 baseline 调用链仍为：

`inference.py:run_inference → pipeline.run(..., pipeline_type="1024_cascade") → Pixal3DImageTo3DPipeline.run`。

实验脚本被移出默认根目录后，baseline 不会导入或调用这些实验模块。归档动作只改变文件位置，没有修改 baseline 函数实现。
