---
name: paddle-debug
description: 在 Paddle 代码库中定位问题并输出高质量调试报告的专用技能。当遇到以下场景时优先使用：(1) Paddle 框架 bug 调试，(2) 算子实现问题排查，(3) 训练脚本异常诊断，(4) 分布式训练故障定位，(5) CUDA/GPU 相关错误处理，(6) 需要生成结构化调试报告。
---

# Paddle 仓库调试

## 调试流程概览

调试遵循以下步骤：

1. 描述问题并构造最小复现
2. 代码定位与多假设验证
3. 先写问题分析报告，再做最小修复
4. 利用 Git / CI 收束和巩固结论

## 步骤 1：描述问题并构造最小复现

用简洁的自然语言说明：
- 触发步骤（命令、脚本、关键配置）
- 期望行为 vs 实际行为
- 是否只在特定环境 / 机器 / 设备 / 数据子集上出现

**先确认 bug 能被稳定复现**。若无法复现：
- 检查命令是否抄错、参数是否缺失
- 比对并对齐环境（Paddle / Python / CUDA / CUDNN / 驱动 / 显卡型号等）
- 确认与最初出问题的环境一致后再继续

抽取独立的 Python 脚本承载问题：
- 固定随机种子（`numpy` / `random` / `paddle.seed` 等）
- 使用固定、可序列化的小数据
- 去掉与问题无关的逻辑

目标：一条命令即可复现 `python reproduce_xxx.py`。

## 步骤 2：代码定位与多假设验证

### 使用工具定位代码

- **ast-grep**：用于结构化代码搜索，快速定位特定代码模式

### 带观测点的复现

阅读报错栈和相关代码时，先列出**多个可能原因假设**（数据异常、shape 错误、数值不稳定、环境不一致、算子实现问题等），不要立刻改代码。

围绕假设在关键路径上加入**观测点**：

| 观测方式 | 用途 |
|---------|------|
| 打印与断言 | 在关键算子调用前后，打印 Tensor 的 shape、dtype、device、数值范围（min/max/mean） |
| 对比法 | 对同一逻辑分别在 CPU / GPU 上运行，比较中间结果差异 |
| 版本与环境信息 | 记录 `paddle.__version__`、CUDA/CUDNN 版本、驱动信息等 |

每完成一次带观测点的复现：
- 基于运行时数据**排除不成立的假设**
- 在更窄的范围内继续加观测点，逐步缩小问题所在的模块 / 算子 / 配置

将调试日志保存到 `.paddle-agent/debug-logs/` 目录。

## 步骤 3：先写问题分析报告，再做最小修复

基于已有观测和对比结果，先完成**问题分析报告**：

```markdown
# [问题标题]

## 复现方式
- 命令：
- 环境：
- 最小脚本路径：

## 现象描述
[错误信息或异常行为]

## 根因分析
[配置 / 数据 / 框架 / 算子 / 环境中的哪一处有问题]

## 关键证据
[日志片段、对比结果、重要观测点输出]
```

报告存放在 `.paddle-agent/debug-analysis/` 目录，没有该目录请创建。

归因时考虑以下维度：
- **接口 / 形状 / dtype**：哪个 Tensor 的 shape / dtype 与预期不符
- **NaN / Inf / 数值发散**：哪一层首次出现异常数值
- **性能与显存**：瓶颈在 CPU、IO 还是 GPU kernel

**只有在分析结论较为充分时**，才进入最小修复阶段：
- 设计改动面尽量小的修改来验证根因
- 先用最小复现脚本验证修复
- 再用完整训练 / 推理脚本验证关键业务路径

## 步骤 4：利用 Git / CI 收束和巩固结论，最后总结保存为文件

当判断问题可能由近期提交引入时：
- 使用 `git bisect` 对可疑提交范围做二分定位

对已定位的问题：
- 补充覆盖最小复现脚本逻辑的单测
- 留意 CI 中相关用例是否出现新增失败
- 将最终结论沉淀到 `.paddle-agent/debug-analysis/`

## CUDA / GPU 调试

**详细的 CUDA 调试技巧**：见 [references/cuda-debug.md](references/cuda-debug.md)

快速参考：
```bash
# 启用错误检查环境变量复现问题
FLAGS_check_cuda_error=1 FLAGS_use_system_allocator=1 python reproduce.py
```

关键点：
- CUDA 错误通常是异步的，使用 `FLAGS_check_cuda_error=1` 让错误立即暴露
- GPU kernel 调用前必须检查 numel/shape 是否为空
- 空 Tensor（numel=0）会导致 grid size=0，触发 CUDA error(9)
- CUDA API 返回值必须全部检查，忽略返回值会导致 sticky error 残留
- `PADDLE_ENFORCE_GPU_SUCCESS` 不会调用 `cudaGetLastError()`，在错误路径上需手动清除

## 注意事项

- 调试的第一目标是**稳定复现并缩小范围**，不要一开始就尝试大规模重构
- 任何「只在某些机器上出现」的问题，优先从环境差异入手
- 在 Paddle 仓库遇到 bug 时，优先按本 skill 流程执行，再考虑具体修复实现

### 算子修复注意事项

- **前向和反向 kernel 要一并检查**：反向 kernel 往往复用相同的计算逻辑，同样存在边界问题
- **检查所有入口函数**：底层公共函数可能被多个入口调用，确保边界检查在正确的层级
- **头文件修改需完整重编**：修改 `.h` 后需重新编译所有引用它的 `.cu`，并重新链接 `.so`

### CUDA API 与 Sticky Error 注意事项

- **所有 CUDA API 返回值必须检查**：包括 `cudaEventSynchronize`、`cudaStreamSynchronize` 等，忽略返回值不仅丢失错误信息，还会导致 CUDA runtime 中残留 sticky error
- **错误路径必须清除 last error**：在 `PADDLE_ENFORCE_GPU_SUCCESS` 抛出异常之前，手动调用 `cudaGetLastError()` 清除残留错误，否则 Python `try/except` 捕获异常后 CUDA 状态仍被污染
- **跨测试状态污染**：unittest 中一个测试的 CUDA sticky error 会影响后续所有测试，排查时需关注测试执行顺序
- **定位 sticky error 污染源**：通过逐步删减测试来二分定位产生残留错误的源头测试

### Paddle 编译验证流程

修改 kernel 头文件后的增量编译：
```bash
cd build
# 编译修改的 kernel
ninja paddle/phi/CMakeFiles/phi_gpu.dir/kernels/gpu/<kernel_name>.cu.o -j512
# 重新链接 phi_gpu
ninja phi_gpu -j512
# 重新链接 libpaddle.so
ninja paddle/fluid/pybind/libpaddle.so -j512
# 如果 Python 库未自动更新，手动复制
cp paddle/fluid/pybind/libpaddle.so python/paddle/base/libpaddle.so
```

## 调试案例

见 [references/case-studies.md](references/case-studies.md) 了解实际调试案例。