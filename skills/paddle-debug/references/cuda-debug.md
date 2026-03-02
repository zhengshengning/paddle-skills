# CUDA / GPU 调试技巧

## 常用调试环境变量

| 环境变量 | 作用 |
|---------|------|
| `FLAGS_check_cuda_error=1` | 启用 CUDA 同步错误检查，将异步错误立即暴露 |
| `FLAGS_use_system_allocator=1` | 使用系统内存分配器，便于排查显存相关问题 |
| `CUDA_LAUNCH_BLOCKING=1` | 强制 CUDA kernel 同步执行，便于定位出错的 kernel |
| `FLAGS_cudnn_exhaustive_search=0` | 关闭 cuDNN 算法搜索，排除算法选择导致的不确定性 |
| `FLAGS_cudnn_deterministic=1` | 启用 cuDNN 确定性模式 |
| `NCCL_DEBUG=INFO` | 启用 NCCL 调试日志（分布式训练问题排查） |

## 常见 CUDA 错误码

| 错误码 | 名称 | 常见原因 |
|--------|------|----------|
| `CUDA error(1)` | cudaErrorInvalidValue | 非法参数值，如空指针、越界 |
| `CUDA error(2)` | cudaErrorMemoryAllocation | 显存分配失败，OOM |
| `CUDA error(4)` | cudaErrorLaunchFailure | kernel 启动失败 |
| `CUDA error(9)` | cudaErrorInvalidConfiguration | kernel 配置无效（grid/block size 为 0 或超限） |
| `CUDA error(11)` | cudaErrorInvalidValue | 非法设备指针或参数 |
| `CUDA error(700)` | cudaErrorIllegalAddress | 非法内存访问 |
| `CUDA error(719)` | cudaErrorLaunchFailure | kernel 执行期间出错 |

## 边界条件检查要点

GPU kernel 对边界条件特别敏感，常见需要检查的场景：

- **空 Tensor / numel=0**：确保在调用 kernel 前检查，避免 grid size 为 0
- **0-d Tensor（标量）**：shape=() 时 numel=1，但处理逻辑可能与高维不同
- **极小 batch**：batch=1 或某个维度为 1 时，可能触发特殊分支
- **极大 shape**：可能导致 int32 溢出或超出 GPU 线程限制
- **数值边界**：NaN、Inf、极大/极小值可能导致数值不稳定

## 调试步骤示例

1. 启用错误检查环境变量复现问题：
   ```bash
   FLAGS_check_cuda_error=1 FLAGS_use_system_allocator=1 python reproduce.py
   ```

2. 分析错误栈，定位到具体 kernel 或 API 调用

3. 检查该路径的边界条件处理：
   - 是否在调用 kernel 前检查了 numel == 0？
   - 是否正确处理了 0-d tensor？
   - grid/block size 计算是否可能为 0？

4. 在关键位置添加日志，打印 shape、numel、配置参数

5. 修复后使用相同环境变量验证