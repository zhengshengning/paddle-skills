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
| `CUDA error(400)` | cudaErrorInvalidResourceHandle | 无效的 CUDA 资源句柄（stream/event 未初始化或已销毁） |
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

## CUDA Sticky Error（残留错误）机制

### 什么是 CUDA Sticky Error

CUDA runtime 维护一个 "last error" 状态。当 CUDA API 调用失败时，错误会被记录到这个状态中。这个残留错误**不会自动清除**，必须通过 `cudaGetLastError()` 显式读取并清除。

如果一个错误未被清除，后续所有依赖 `cudaGetLastError()` 检查的操作都会检测到它——即使后续操作本身没有问题。

### 为什么这在 Paddle 中特别重要

Paddle 的 `FLAGS_check_cuda_error=1` 模式下，每个算子前后都会调用：
```cpp
void CUDAErrorCheck(const std::string& check_tag) {
    cudaDeviceSynchronize();   // 同步所有 pending 操作
    cudaGetLastError();        // 检查并清除 last error → 如果非0则抛异常
}
```
因此，之前任何未清除的 CUDA error 都会在下一个算子执行时被检测到，表现为"看似无关的代码报错"。

### 常见 Sticky Error 产生场景

| 场景 | 原因 | 表现 |
|------|------|------|
| 未检查的 CUDA API 返回值 | 忽略了 `cudaEventSynchronize` 等 API 的返回值 | 错误残留，后续操作失败 |
| try/catch 捕获 CUDA 异常 | C++ 异常被捕获但未调用 `cudaGetLastError()` | 异常虽被处理，错误仍残留 |
| 异步 kernel 失败 | kernel 在异步执行中出错，直到同步时才被发现 | 错误在首次同步点被报告 |

### 防御编码规范

```cpp
// 错误示例：忽略返回值
cudaEventSynchronize(event);  // ← 返回值丢弃，错误残留

// 正确示例 1：使用 PADDLE_ENFORCE_GPU_SUCCESS 检查
PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(event));

// 正确示例 2：手动检查并清除（适用于允许失败的场景）
gpuError_t err = cudaEventSynchronize(event);
if (err != cudaSuccess) {
    cudaGetLastError();  // 清除 last error
    // 进行错误处理...
}

// 注意：PADDLE_ENFORCE_GPU_SUCCESS 不会调用 cudaGetLastError()，
// 如果需要在抛异常前清除 last error，必须手动调用。
```

### 调试 Sticky Error 的方法

1. **确认是否为残留错误**：在报错点之前手动插入 `cudaGetLastError()` 调用，如果能清除错误且后续正常，则说明是残留
2. **二分测试定位污染源**：逐步删减之前执行的测试/操作，找到产生残留错误的源头
3. **Python 层快速验证**：
   ```python
   import ctypes
   libcudart = ctypes.CDLL("libcudart.so")
   err = libcudart.cudaGetLastError()
   print(f"CUDA last error: {err}")  # 0 表示无错误
   ```