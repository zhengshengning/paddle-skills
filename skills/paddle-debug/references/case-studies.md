# 调试案例

## 案例：one_hot kernel CUDA error(9) 修复

### 问题描述
```bash
FLAGS_check_cuda_error=1 FLAGS_use_system_allocator=1 python test_one_hot_v2_op.py
```
报错：`CUDA error(9), invalid configuration argument`

### 根因分析
1. 测试用例 `TestOneHotOp_ZeroSize` 使用 `x_shape=[0, 10, 7, 3]`，即 `numel=0`
2. `one_hot_kernel.cu` 中代码顺序：
   ```cpp
   funcs::set_constant(dev_ctx, out, 0.0);  // 先调用 kernel
   if (numel == 0) return;                   // 后检查边界
   ```
3. `set_constant` 内部启动 CUDA kernel，numel=0 导致 grid size=0，触发 CUDA error(9)

### 修复方案
将边界检查移到 kernel 调用之前：
```cpp
if (numel == 0) return;  // 先检查边界
funcs::set_constant(dev_ctx, out, 0.0);  // 后调用 kernel
```

### 修复文件
- `paddle/phi/kernels/gpu/one_hot_kernel.cu`
- `paddle/phi/kernels/legacy/gpu/one_hot_kernel.cu`

### 经验总结
- GPU kernel 调用前必须检查 numel/shape 是否为空
- 同一算子可能有多个实现（主版本和 legacy），需同步修复
- 使用 `FLAGS_check_cuda_error=1` 可以将异步 CUDA 错误立即暴露