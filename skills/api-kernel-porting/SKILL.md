---
name: api-kernel-porting
description: |
  辅助将 PyTorch 等外部仓库的 Kernel 实现移植到 PaddlePaddle。
  包含代码检索、自动生成、编译验证及单元测试全流程。
---

# API Kernel 代码移植助手

本 Skill 旨在帮助开发者将外部框架（如 PyTorch）的 Kernel 实现快速、准确地移植到 PaddlePaddle 仓库中。

## 核心能力

1.  **参考代码检索**：根据算子名称自动在参考仓库（如 `src/pytorch`）中查找对应的 Kernel 实现。
2.  **代码移植生成**：基于参考实现，生成符合 Paddle PHI 架构规范的 C++/CUDA Kernel 代码。
3.  **自动化编译**：调用构建系统进行增量编译，验证代码语法的正确性。
4.  **功能验证**：编写并运行 Python 单元测试，确保移植后的算子行为符合预期，使得paddle.xxx与torch.xxx计算结果逐位对齐。

## 工作流程

### 1. 信息收集与检索 (Context Gathering)

当用户请求移植某个 API 或 Kernel 时：

1.  **获取目标信息**：确认 Paddle 侧的目标路径（通常在 `src/Paddle/paddle/phi/kernels` 下）和 API 接口定义。
2.  **检索参考实现**：
    *   如果用户未提供具体文件路径，使用 `search_files` 在 `src/pytorch` (或其他参考库) 中搜索相关关键字（如 `filename.cu`, `kernel_name`）。
    *   重点关注 `aten/src/ATen/native` 目录下的实现。
3.  **读取上下文**：使用 `extract_content_blocks` 读取参考代码和 Paddle 侧相关的头文件或现有实现。

### 2. 代码生成 (Code Generation)

基于读取的上下文，生成 Paddle Kernel 代码，请尽可能严格参照torch代码实现和代码结构，以确保最后移植的正确性。如果paddle已经存在相应的Kernel代码实现，请保留原来的实现，生成一个新的Kernel名字，并将原Kernel注册'PD_REGISTER_KERNEL(topk'改为'PD_REGISTER_KERNEL(topk_old'，将新增加的Kernel注册'PD_REGISTER_KERNEL(topk'。

注意以下差异：

*   **数据结构**：Torch 使用 `Tensor` / `TensorAccessor`，Paddle PHI 使用 `phi::DenseTensor`。
*   **上下文管理**：Paddle 使用 `phi::DeviceContext`。
*   **分配器**：Paddle 内存分配通常通过 Context 进行。
*   **CUDA 核函数**：确保线程块配置和 launch 逻辑适配 Paddle 的宏（如 `PD_DISPATCH_FLOATING_AND_HALF_TYPES`）。



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

## 交互示例

**Python虚拟环境**: source paddle_env/bin/activate 
**用户**: "请把 PyTorch 的 `softplus` CUDA kernel 移植到 Paddle，目标是 `paddle/phi/kernels/gpu/softplus_kernel.cu`。"

**Agent**:
1.  **搜索**: `search_files` 搜索 `softplus` 相关 `.cu` 文件，找到 `src/pytorch/aten/src/ATen/native/cuda/Activation.cu` 等。
2.  **读取**: 读取 Torch 实现代码。
3.  **生成**: 编写 `src/Paddle/paddle/phi/kernels/gpu/softplus_kernel.cu`，适配 PHI 接口。
4.  **编译**: 运行 `make -j 512`。
5.  **测试**: 编写 `test_softplus.py`，比较 `paddle.nn.functional.softplus` 和 `torch.nn.functional.softplus` 的结果，要求两者计算结果逐位对齐。
