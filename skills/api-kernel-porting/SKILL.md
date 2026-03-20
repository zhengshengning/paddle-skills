---
name: api-kernel-porting
description: |
  辅助将 PyTorch 等外部仓库的 Kernel 实现移植到 PaddlePaddle。
  包含代码检索、自动生成、编译验证、功能验证及性能测试全流程。
---

# API Kernel 代码移植助手

本 Skill 旨在帮助开发者将外部框架（如 PyTorch）的 Kernel 实现快速、准确地移植到 PaddlePaddle 仓库中，同时**确保移植后的代码性能与原实现相当**。

## 核心原则

> **最小修改原则**：移植过程中应尽可能保持原始代码的实现逻辑、算法结构和优化策略不变，仅做必要的接口适配。这是保证移植后无性能下降的关键。

## 核心能力

1.  **参考代码检索**：根据算子名称自动在参考仓库（如 `src/pytorch`）中查找对应的 Kernel 实现。
2.  **代码移植生成**：基于参考实现，生成符合 Paddle PHI 架构规范的 C++/CUDA Kernel 代码，**严格保持原有算法实现和优化策略**。
3.  **自动化编译**：调用构建系统进行增量编译，验证代码语法的正确性。
4.  **功能验证**：编写并运行 Python 单元测试，确保移植后的算子行为符合预期，使得paddle.xxx与torch.xxx计算结果逐位对齐。
5.  **性能测试**：对比 Paddle 移植实现与原始 PyTorch 实现的执行性能，确保无性能退化。

## 工作流程

### 1. 信息收集与检索 (Context Gathering)

当用户请求移植某个 API 或 Kernel 时：

1.  **获取目标信息**：确认 Paddle 侧的目标路径（通常在 `src/Paddle/paddle/phi/kernels` 下）和 API 接口定义。
2.  **检索参考实现**：
    *   如果用户未提供具体文件路径，使用 `search_files` 在 `src/pytorch` (或其他参考库) 中搜索相关关键字（如 `filename.cu`, `kernel_name`）。
    *   重点关注 `aten/src/ATen/native` 目录下的实现。
3.  **读取上下文**：使用 `extract_content_blocks` 读取参考代码和 Paddle 侧相关的头文件或现有实现。

### 2. 代码生成 (Code Generation)

基于读取的上下文，生成 Paddle Kernel 代码。**移植的核心原则是保持原始实现不变，仅做接口适配**。

#### 2.1 移植原则（性能保障）

为确保移植后无性能下降，必须遵循以下原则：

1.  **算法逻辑完全保持**：
    *   不修改原有的计算逻辑和数学公式
    *   不改变循环结构、分支条件
    *   不"优化"或"简化"原有代码逻辑

2.  **CUDA 优化策略保持**：
    *   **线程配置保持**：保留原有的 `block_size`、`grid_size` 计算方式
    *   **内存访问模式保持**：保留原有的 coalesced access、shared memory 使用策略
    *   **循环展开保持**：保留 `#pragma unroll` 等优化指令
    *   **向量化访问保持**：如原代码使用 `float4`、`int4` 等向量类型，必须保留

3.  **模板和特化保持**：
    *   保留原有的模板参数设计
    *   保留所有的模板特化版本
    *   保留编译期常量优化（如 `if constexpr`）

4.  **仅做必要的接口适配**：
    *   数据结构类型替换（如 `at::Tensor` → `phi::DenseTensor`）
    *   设备上下文获取方式替换
    *   内存分配接口替换
    *   类型分发宏替换

#### 2.2 接口适配映射表

| PyTorch | Paddle PHI | 说明 |
|---------|------------|------|
| `at::Tensor` / `Tensor` | `phi::DenseTensor` | 张量类型 |
| `TensorAccessor<T, N>` | `phi::DenseTensor::data<T>()` | 数据访问 |
| `at::cuda::getCurrentCUDAStream()` | `ctx.stream()` | CUDA 流 |
| `c10::cuda::CUDACachingAllocator` | `phi::memory_utils` | 内存分配 |
| `AT_DISPATCH_FLOATING_TYPES` | `PD_DISPATCH_FLOATING_TYPES` | 类型分发 |
| `TORCH_CHECK` | `PADDLE_ENFORCE` | 断言检查 |

#### 2.3 处理已存在的实现

如果 Paddle 已经存在相应的 Kernel 代码实现，请保留原来的实现作为备份：

1.  将原 Kernel 注册改为旧版本：`PD_REGISTER_KERNEL(topk` → `PD_REGISTER_KERNEL(topk_old`
2.  新增的移植 Kernel 使用原名注册：`PD_REGISTER_KERNEL(topk`
3.  便于后续性能对比和回滚



### 3. 编译验证 (Compilation)

移植代码写入后，必须进行编译验证。请在 `src/Paddle/build` 目录下执行编译命令。

**参考编译命令**：

```bash
# 切换到构建目录
cd /root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/src/Paddle/build

# 配置 CMake (通常只需运行一次，若已配置可跳过)
cmake .. -DON_INFER=OFF -DPY_VERSION=3.10 -DWITH_GLOO=OFF -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_DISTRIBUTE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 执行并行编译
make -j 512
```

*提示：`make` 会自动处理增量编译。如果遇到编译错误，分析错误日志并修正代码，直到编译成功。*

### 4. 单元测试与验证 (Verification)

编译成功后，进行功能验证：

1.  **更新环境**：
    *   如果修改了 C++ 核心库，可能需要重新安装 whl 包：
        ```bash
        cd /root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/src/Paddle/build/python/dist
        pip uninstall -y paddlepaddle-gpu
        pip install *.whl
        ```
2.  **编写测试脚本**：
    *   在合适的位置（如 `src/Paddle/python/paddle/fluid/tests/unittests/` 或用户指定目录）创建 Python 测试脚本。
    *   **对比测试**：构造相同的 Input，分别调用 Paddle 新实现的 API 和 PyTorch 原生 API（或 Numpy 实现），验证 Output 是否一致（注意精度误差 `atol`, `rtol`）。
3.  **运行测试**：
    *   执行 `python path/to/test_script.py` 并解析结果。

### 5. 性能测试与验证 (Performance Testing)

功能验证通过后，**必须进行性能测试**，确保移植后的实现与原始 PyTorch 实现性能相当。

#### 5.1 性能测试脚本模板

```python
import time
import numpy as np
import paddle
import torch

def benchmark_kernel(framework, func, input_tensors, warmup=10, repeat=100):
    """
    通用性能测试函数
    Args:
        framework: 'paddle' 或 'torch'
        func: 待测试的函数
        input_tensors: 输入张量列表
        warmup: 预热次数
        repeat: 测试重复次数
    Returns:
        平均执行时间 (ms)
    """
    # GPU 同步
    if framework == 'paddle':
        sync_fn = paddle.device.cuda.synchronize
    else:
        sync_fn = torch.cuda.synchronize
    
    # 预热
    for _ in range(warmup):
        _ = func(*input_tensors)
        sync_fn()
    
    # 正式测试
    sync_fn()
    start = time.perf_counter()
    for _ in range(repeat):
        _ = func(*input_tensors)
    sync_fn()
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000  # 转换为 ms
    avg_time_ms = total_time_ms / repeat  # 计算平均时间
    
    return avg_time_ms


def run_performance_comparison(op_name, paddle_func, torch_func, test_shapes, dtypes=[np.float32]):
    """
    运行 Paddle vs PyTorch 性能对比
    """
    print(f"\n{'='*60}")
    print(f"Performance Comparison: {op_name}")
    print(f"{'='*60}")
    
    results = []
    for shape in test_shapes:
        for dtype in dtypes:
            # 创建测试数据
            np_data = np.random.randn(*shape).astype(dtype)
            
            paddle_tensor = paddle.to_tensor(np_data, place=paddle.CUDAPlace(0))
            torch_tensor = torch.tensor(np_data, device='cuda')
            
            # Paddle 性能测试
            paddle_time = benchmark_kernel(
                'paddle', paddle_func, [paddle_tensor]
            )
            
            # PyTorch 性能测试
            torch_time = benchmark_kernel(
                'torch', torch_func, [torch_tensor]
            )
            
            # 计算性能比率
            ratio = paddle_time / torch_time
            status = "✓ PASS" if ratio <= 1.1 else "✗ FAIL"  # 允许 10% 误差
            
            print(f"\nShape: {shape}, dtype: {dtype}")
            print(f"  Paddle: {paddle_time:.4f} ms")
            print(f"  PyTorch: {torch_time:.4f} ms")
            print(f"  Ratio (Paddle/PyTorch): {ratio:.4f} {status}")
            
            results.append({
                'shape': shape,
                'dtype': dtype,
                'paddle_ms': paddle_time,
                'torch_ms': torch_time,
                'ratio': ratio,
                'pass': ratio <= 1.1
            })
    
    # 总结
    print(f"\n{'='*60}")
    passed = sum(1 for r in results if r['pass'])
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n⚠️ WARNING: Performance regression detected!")
        print("请检查以下方面：")
        print("  1. 是否保持了原有的 CUDA 线程配置？")
        print("  2. 是否保持了原有的内存访问模式？")
        print("  3. 是否遗漏了重要的优化指令？")
    
    return all(r['pass'] for r in results)


# 使用示例
if __name__ == "__main__":
    # 定义测试规模（覆盖小、中、大规模）
    test_shapes = [
        (64, 64),
        (256, 256),
        (1024, 1024),
        (4096, 4096),
        (16, 3, 224, 224),   # CV 典型输入
        (32, 512, 768),      # NLP 典型输入
    ]
    
    # 示例：测试 softplus 算子
    passed = run_performance_comparison(
        op_name="softplus",
        paddle_func=paddle.nn.functional.softplus,
        torch_func=torch.nn.functional.softplus,
        test_shapes=test_shapes,
        dtypes=[np.float32, np.float16]
    )
    
    assert passed, "Performance test failed!"
```

#### 5.2 性能测试要求

1.  **测试规模覆盖**：
    *   小规模：`(64, 64)` ~ `(256, 256)`
    *   中规模：`(1024, 1024)` ~ `(4096, 4096)`
    *   大规模：视具体算子场景而定
    *   典型场景：CV `(N, C, H, W)`，NLP `(B, S, H)`

2.  **数据类型覆盖**：
    *   `float32` (必测)
    *   `float16` (如支持)
    *   `bfloat16` (如支持)

3.  **性能指标要求**：
    *   **Paddle/PyTorch 耗时比 ≤ 1.1**（即 Paddle 耗时不超过 PyTorch 的 110%）
    *   如超出阈值，需排查移植过程中是否改变了关键优化

4.  **测试环境一致性**：
    *   确保 Paddle 和 PyTorch 使用相同的 GPU
    *   确保测试前 GPU 处于空闲状态
    *   使用足够的 warmup 消除首次运行开销

#### 5.3 性能问题排查

如果性能测试未通过，请按以下顺序排查：

1.  **线程配置检查**：对比 `block_size`、`grid_size` 是否与原实现一致
2.  **内存访问检查**：是否破坏了 coalesced memory access
3.  **优化指令检查**：是否遗漏 `#pragma unroll`、`__restrict__` 等
4.  **分支发散检查**：是否引入了新的条件分支导致 warp divergence
5.  **共享内存检查**：shared memory 的使用是否与原实现一致

## 交互示例

**Python虚拟环境**: source paddle_env/bin/activate 
**用户**: "请把 PyTorch 的 `softplus` CUDA kernel 移植到 Paddle，目标是 `paddle/phi/kernels/gpu/softplus_kernel.cu`。"

**Agent**:
1.  **搜索**: `search_files` 搜索 `softplus` 相关 `.cu` 文件，找到 `src/pytorch/aten/src/ATen/native/cuda/Activation.cu` 等。
2.  **读取**: 读取 Torch 实现代码，分析其算法逻辑、CUDA 优化策略（线程配置、内存访问模式等）。
3.  **生成**: 编写 `src/Paddle/paddle/phi/kernels/gpu/softplus_kernel.cu`，**严格保持原有实现逻辑**，仅做接口适配。
4.  **编译**: 运行 `make -j 512`。
5.  **功能测试**: 编写 `test_softplus.py`，比较 `paddle.nn.functional.softplus` 和 `torch.nn.functional.softplus` 的结果，要求两者计算结果逐位对齐。
6.  **性能测试**: 运行性能对比脚本，验证 Paddle 实现的执行时间不超过 PyTorch 的 110%，确保无性能退化。

## 完整验收标准

移植工作完成需满足以下全部条件：

1.  ✅ 编译通过，无错误警告
2.  ✅ 功能测试通过，计算结果逐位对齐
3.  ✅ 性能测试通过，Paddle/PyTorch 耗时比 ≤ 1.1
