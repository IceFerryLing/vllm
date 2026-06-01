# vLLM 剪裁版本

本仓库是基于 **vLLM 0.18.1** 的剪裁版本，保留了当前项目所需的核心代码与功能。

## `csrc/` 和 `vllm/` 目录结构说明

下面用带注释的文件树说明两个核心目录的作用。`csrc/` 主要是 CUDA/C++ 原生扩展和高性能算子，`vllm/` 主要是 Python 运行时、服务入口、模型执行和分布式调度逻辑。

### `csrc/`

```text
csrc/
├── attention/                         # GPU 注意力相关 kernel，包含 PagedAttention、MLA 和注意力状态合并
│   └── mla/                           # Multi-head Latent Attention 相关实现
│       └── cutlass_sm100_mla/         # 基于 CUTLASS、面向 SM100 架构的 MLA kernel
│           ├── device/                # device 侧封装和调度接口
│           └── kernel/                # 具体 CUDA/CUTLASS kernel 实现
├── core/                              # C++ 扩展通用基础设施，例如异常、类型、数学工具和注册机制
├── cpu/                               # CPU 后端算子实现，覆盖注意力、LayerNorm、MoE、位置编码等
│   ├── micro_gemm/                    # CPU 小矩阵乘法 kernel
│   └── sgl-kernels/                   # CPU 侧辅助 kernel 或第三方集成 kernel
├── cutlass_extensions/                # CUTLASS 扩展工具，服务于 GEMM、MoE、量化等高性能 CUDA 算子
│   └── epilogue/                      # GEMM epilogue 后处理逻辑，例如缩放、激活、类型转换
├── mamba/                             # Mamba/SSM 模型相关原生算子
│   └── mamba_ssm/                     # selective scan 等 SSM 计算核心
├── moe/                               # Mixture-of-Experts 原生算子，包含路由、Top-K、专家重排和 MoE GEMM
│   ├── marlin_moe_wna16/              # Marlin 风格的 MoE WNA16 量化实现
│   ├── mxfp8_moe/                     # MXFP8 MoE kernel
│   └── permute_unpermute_kernels/     # MoE token 重排和还原 kernel
├── quantization/                      # 量化相关 CUDA/C++ 实现
│   ├── awq/                           # AWQ 量化 kernel
│   ├── cutlass_w4a8/                  # CUTLASS W4A8 量化矩阵乘
│   ├── fp4/                           # FP4 量化支持
│   ├── fused_kernels/                 # 量化场景下的融合 kernel
│   ├── gguf/                          # GGUF 权重量化格式相关支持
│   ├── gptq/                          # GPTQ 量化 kernel
│   ├── gptq_allspark/                 # AllSpark GPTQ 相关实现
│   ├── hadamard/                      # Hadamard 变换相关量化工具
│   │   └── hadacore/                  # Hadamard 核心实现
│   ├── machete/                       # Machete 量化矩阵乘实现
│   ├── marlin/                        # Marlin 量化 kernel
│   └── w8a8/                          # W8A8 量化实现
│       ├── cutlass/                   # 基于 CUTLASS 的 W8A8 kernel
│       │   ├── c3x/                   # CUTLASS 3.x 相关实现
│       │   └── moe/                   # W8A8 MoE kernel
│       ├── fp8/                       # FP8 W8A8 支持
│       │   ├── amd/                   # AMD/ROCm FP8 实现
│       │   └── nvidia/                # NVIDIA FP8 实现
│       └── int8/                      # INT8 W8A8 实现
├── quickreduce/                       # 快速规约和自定义 reduce 通信相关实现
├── rocm/                              # AMD ROCm 后端原生算子和 PyTorch 绑定
└── sparse/                            # 稀疏计算相关 kernel
    └── cutlass/                       # 基于 CUTLASS 的稀疏矩阵计算实现
```

`csrc/` 根目录下还有一批直接放置的 CUDA/C++ 文件，它们通常是会被 Python 侧直接调用的核心算子或扩展入口：

```text
csrc/
├── torch_bindings.cpp                 # 主 CUDA/C++ 扩展的 PyTorch 绑定入口
├── ops.h                              # 扩展算子声明和导出接口
├── activation_kernels.cu              # 激活函数 kernel
├── cache*.cu, cache.h                 # KV cache 读写、复制、重排和融合 cache kernel
├── layernorm*_kernels.cu              # LayerNorm、RMSNorm 及量化相关归一化 kernel
├── pos_encoding_kernels.cu            # 位置编码和 RoPE 相关 kernel
├── sampler.cu, topk.cu                # 采样和 Top-K 选择 kernel
├── custom_all_reduce*.cu/.cuh         # 自定义 AllReduce 通信 kernel
├── custom_quickreduce.cu              # QuickReduce 相关 CUDA 入口
├── cumem_allocator.*                  # CUDA 内存分配器封装
├── cuda_*                             # CUDA 兼容层、工具函数、向量工具和 view 辅助代码
├── dsv3_fused_a_gemm.cu               # DeepSeek V3 相关融合 GEMM
├── fused_qknorm_rope_kernel.cu        # QK Norm 和 RoPE 融合 kernel
├── permute_cols.cu                    # 列重排工具 kernel
└── type_convert.cuh                   # CUDA 类型转换工具
```

### `vllm/`

```text
vllm/
├── assets/                            # 多模态测试和示例资源的抽象，包含图片、音频、视频资源处理
├── benchmarks/                        # 基准测试脚本和数据集工具
│   ├── lib/                           # benchmark 共用工具
│   └── sweep/                         # 参数扫描和批量 benchmark 支持
├── compilation/                       # torch.compile、CUDA Graph 和静态图编译相关逻辑
│   └── passes/                        # 编译图优化 pass
│       ├── fusion/                    # 融合类图优化
│       └── utility/                   # 图变换辅助 pass
├── config/                            # vLLM 配置体系，按模型、缓存、并行、调度、LoRA、量化等拆分
├── device_allocator/                  # 设备内存分配器封装，例如 CUDA cuMem allocator
├── distributed/                       # 分布式推理、通信、并行状态和跨节点数据传输
│   ├── device_communicators/          # GPU/CPU/XPU 等设备通信器
│   ├── ec_transfer/                   # expert cache 或 elastic cache 相关传输逻辑
│   │   └── ec_connector/              # EC 传输连接器
│   ├── elastic_ep/                    # 弹性 expert parallel 相关逻辑
│   ├── eplb/                          # expert parallel load balancing，专家并行负载均衡
│   │   └── policy/                    # EPLB 策略实现
│   ├── kv_transfer/                   # KV cache 在 prefill/decode 或不同 worker 间的传输
│   │   └── kv_connector/              # KV 传输连接器
│   │       └── v1/                    # v1 执行路径使用的 KV connector
│   └── weight_transfer/               # 权重传输和动态加载相关逻辑
├── engine/                            # 旧版 LLMEngine/AsyncLLMEngine，请求调度和引擎协议
├── entrypoints/                       # 对外入口：CLI、OpenAI API、Anthropic API、serve、gRPC 等
│   ├── anthropic/                     # Anthropic 风格 API 兼容入口
│   ├── cli/                           # 命令行入口
│   │   └── benchmark/                 # CLI benchmark 子命令
│   ├── mcp/                           # MCP 服务入口
│   ├── openai/                        # OpenAI 兼容 API 服务
│   │   ├── chat_completion/           # Chat Completions 接口
│   │   ├── completion/                # Completions 接口
│   │   ├── engine/                    # OpenAI 服务侧 engine 封装
│   │   ├── generate/                  # generate 接口封装
│   │   ├── models/                    # models 接口
│   │   ├── parser/                    # 请求解析
│   │   ├── realtime/                  # Realtime API 相关逻辑
│   │   ├── responses/                 # Responses API 相关逻辑
│   │   └── speech_to_text/            # 语音转文本接口
│   ├── pooling/                       # embedding、score、classify 等池化类任务入口
│   │   ├── base/                      # pooling 基础协议
│   │   ├── classify/                  # 分类任务入口
│   │   ├── embed/                     # embedding 任务入口
│   │   ├── pooling/                   # 通用 pooling 任务入口
│   │   └── score/                     # rerank/score 任务入口
│   ├── sagemaker/                     # AWS SageMaker 部署入口
│   └── serve/                         # vllm serve 相关子命令和服务扩展
│       ├── cache/                     # cache 管理接口
│       ├── disagg/                    # prefill/decode 分离部署支持
│       ├── elastic_ep/                # 弹性 EP 服务控制
│       ├── instrumentator/            # metrics 和监控指标导出
│       │   └── static/                # 监控页面静态资源
│       ├── lora/                      # LoRA 管理接口
│       ├── profile/                   # profile 接口
│       ├── render/                    # prompt/template 渲染接口
│       ├── rlhf/                      # RLHF 相关服务接口
│       ├── rpc/                       # RPC 服务封装
│       ├── sleep/                     # sleep/wake 相关接口
│       └── tokenize/                  # tokenize/detokenize 接口
├── inputs/                            # 输入数据结构、解析和预处理
├── kernels/                           # Python 侧 kernel 封装和 Helion kernel
│   └── helion/                        # Helion kernel 集成
│       ├── configs/                   # Helion kernel 调参配置
│       │   └── silu_mul_fp8/          # silu_mul_fp8 kernel 配置
│       └── ops/                       # Helion ops 封装
├── logging_utils/                     # 日志格式化、访问日志过滤、延迟日志和输入 dump 工具
├── lora/                              # LoRA 权重加载、管理、请求绑定和执行层封装
│   ├── layers/                        # LoRA 版模型层
│   ├── ops/                           # LoRA 算子
│   │   ├── torch_ops/                 # PyTorch 实现
│   │   ├── triton_ops/                # Triton 实现
│   │   └── xpu_ops/                   # XPU 实现
│   └── punica_wrapper/                # Punica LoRA kernel 封装
├── model_executor/                    # 模型执行核心：模型定义、层、权重加载、kernel、offload、warmup
│   ├── kernels/                       # 模型执行用 Python/CUDA kernel 封装
│   │   └── linear/                    # linear 层相关 kernel
│   │       ├── mixed_precision/       # 混合精度 linear
│   │       └── scaled_mm/             # scaled matrix multiplication
│   ├── layers/                        # 模型层实现
│   │   ├── attention/                 # Attention 层和 backend 选择
│   │   ├── fla/                       # FLA 相关层
│   │   │   └── ops/                   # FLA ops
│   │   ├── fused_moe/                 # 融合 MoE 层
│   │   │   ├── configs/               # MoE kernel 调优配置
│   │   │   ├── experts/               # expert 执行实现
│   │   │   ├── oracle/                # MoE oracle 或策略辅助
│   │   │   ├── prepare_finalize/      # MoE 前处理和后处理
│   │   │   ├── router/                # MoE router
│   │   │   └── runner/                # MoE runner
│   │   ├── mamba/                     # Mamba 层
│   │   │   └── ops/                   # Mamba ops
│   │   ├── pooler/                    # pooling 层
│   │   │   ├── seqwise/               # 序列级 pooling
│   │   │   └── tokwise/               # token 级 pooling
│   │   ├── quantization/              # 量化层和量化方法适配
│   │   │   ├── compressed_tensors/    # compressed-tensors 量化格式支持
│   │   │   ├── quark/                 # AMD Quark 量化支持
│   │   │   └── utils/                 # 量化工具
│   │   └── rotary_embedding/          # RoPE/rotary embedding 实现
│   ├── model_loader/                  # 模型权重加载、格式识别和 reload
│   │   └── reload/                    # 动态 reload 支持
│   ├── models/                        # 具体模型架构实现
│   │   └── transformers/              # Hugging Face Transformers 兼容模型封装
│   ├── offloader/                     # 权重或 KV 等 offload 支持
│   └── warmup/                        # 模型 warmup 逻辑
├── multimodal/                        # 多模态输入处理，包含音频、图片、视频、缓存和 processor
│   ├── media/                         # 多模态媒体对象
│   └── processing/                    # 多模态 processor 处理逻辑
├── parser/                            # 通用解析器框架和模型专用解析器
├── platforms/                         # 硬件平台适配层，封装 CUDA、ROCm、CPU、TPU、XPU 等差异
├── plugins/                           # 插件机制
│   ├── io_processors/                 # 输入输出处理插件
│   └── lora_resolvers/                # LoRA 路径和来源解析插件
├── profiler/                          # 性能分析和 layerwise profile 工具
├── ray/                               # Ray 集群和懒加载工具
├── reasoning/                         # reasoning_content 解析器，适配不同模型的思考过程输出格式
├── renderers/                         # prompt、chat template 和多模态输入渲染
│   └── inputs/                        # renderer 输入结构
├── third_party/                       # 内置第三方兼容代码
│   └── flashmla/                      # FlashMLA 相关第三方代码
├── tokenizers/                        # tokenizer 适配层和模型专用 tokenizer
├── tool_parsers/                      # function/tool calling 输出解析器，适配不同模型格式
├── tracing/                           # OpenTelemetry tracing 和链路追踪工具
├── transformers_utils/                # Hugging Face Transformers 集成工具
│   ├── chat_templates/                # chat template 文件或辅助逻辑
│   ├── configs/                       # 模型配置适配
│   │   └── speculators/               # speculative decoding 相关配置
│   └── processors/                    # processor 适配
├── triton_utils/                      # Triton 相关导入、内存分配和工具函数
├── usage/                             # 使用量统计和 usage 上报逻辑
├── utils/                             # 通用工具函数集合
├── v1/                                # vLLM v1 执行路径，包含新调度器、worker、采样和 KV cache 管理
│   ├── attention/                     # v1 attention 抽象和实现
│   │   ├── backends/                  # attention backend
│   │   │   └── mla/                   # MLA backend
│   │   └── ops/                       # attention ops
│   ├── core/                          # v1 核心调度和状态管理
│   │   └── sched/                     # scheduler 实现
│   ├── engine/                        # v1 engine
│   ├── executor/                      # v1 executor，负责调度 worker 执行
│   ├── kv_offload/                    # v1 KV cache offload
│   │   ├── backends/                  # KV offload backend
│   │   └── worker/                    # KV offload worker
│   ├── metrics/                       # v1 metrics
│   ├── pool/                          # v1 pooling 任务支持
│   ├── sample/                        # v1 采样逻辑
│   │   ├── logits_processor/          # logits processor
│   │   └── ops/                       # 采样 ops
│   ├── spec_decode/                   # speculative decoding
│   ├── structured_output/             # 结构化输出约束和状态机
│   └── worker/                        # v1 worker
│       └── gpu/                       # GPU worker
│           ├── metrics/               # GPU worker 指标
│           ├── mm/                    # GPU worker 多模态处理
│           ├── model_states/          # 模型状态管理
│           ├── pool/                  # pooling worker
│           ├── sample/                # GPU 采样
│           └── spec_decode/           # GPU speculative decoding
└── vllm_flash_attn/                   # FlashAttention Python 接口封装
```

`vllm/` 根目录下还有一些重要的单文件模块：

```text
vllm/
├── __init__.py                        # Python 包初始化
├── _custom_ops.py                     # Python 侧访问 csrc 自定义 CUDA/C++ ops 的封装
├── _aiter_ops.py, _oink_ops.py        # 特定后端或实验性 ops 封装
├── _xpu_ops.py                        # XPU ops 封装
├── beam_search.py                     # beam search 实现
├── collect_env.py                     # 环境信息采集脚本
├── connections.py                     # 连接管理辅助代码
├── envs.py, env_override.py           # 环境变量定义和覆盖逻辑
├── exceptions.py                      # vLLM 自定义异常
├── forward_context.py                 # forward 执行上下文
├── logger.py                          # 基础 logger 配置
├── logits_process.py                  # logits 处理逻辑
├── logprobs.py                        # logprob 计算和结构封装
├── model_inspection.py                # 模型结构和能力检查
├── outputs.py                         # 推理输出数据结构
├── pooling_params.py                  # pooling 任务参数
├── sampling_params.py                 # 文本生成采样参数
├── scalar_type.py                     # 标量类型定义
├── scripts.py                         # 脚本入口辅助
├── sequence.py                        # 序列、请求和 token 状态结构
├── tasks.py                           # 任务类型定义
└── version.py                         # 版本信息
```
