---
name: turing-design
description: |
  Turing 算子设计文档生成技能。基于 design.md 标准模板，通过交互问答收集
  算子需求、功能、约束、Tiling、Kernel 和测试信息，自动生成完整的算子
  设计文档。用户提及"生成设计文档"、"写设计文档"、"生成 DESIGN.md"、
  "按 design.md 模板填写算子设计"或要求为某个算子补齐设计文档时使用。
---

# turing-design

生成标准化算子设计文档（`DESIGN.md`）和可选的 aclnn 接口文档。

## 适用场景

使用本技能处理以下请求：

- 用户要求生成、补齐或改写算子设计文档。
- 用户要求按 `design.md` 或 Turing 算子模板填写设计。
- 用户给出算子源码目录，希望从实现反推出需求、接口、Tiling 和测试策略。
- 用户要求补充 aclnn 接口说明文档。

## 总原则

- 先读源码、已有文档、测试和绑定层，再生成文档；已有实现是事实依据。
- 不臆造无法从仓库或用户输入确认的信息。镜像、CMC、客户、性能目标等无法确认时，写明"按交付环境补齐"或向用户追问。
- 优先使用华为内源链接或项目内路径作为参考，不主动引入外网链接。
- 文档必须能被评审：接口、shape、dtype、约束、tiling、workspace、数据流和测试策略要落到具体字段。
- 若发现源码注释、OpDef、kernel 访问布局之间不一致，在文档中明确区分"接口约定"与"当前实现访问方式"，不要把不一致包装成确定事实。

## 工作流

### Step 1: 收集算子基本信息

先确认或从源码提取：

| 问题 | 说明 |
|------|------|
| 算子名称 | 英文名，如 `SwiGLUQuant` |
| 算子功能描述 | 如 "swiglu + int8 量化" |
| 计算公式 | 数学公式、分块公式、特殊分支 |
| 输入参数 | tensor 名称、dtype、shape、含义 |
| 输出参数 | tensor 名称、dtype、shape、含义 |
| 属性参数 | attr 名称、类型、默认值、值域 |
| 目标芯片 | 如 Ascend910B、Ascend910_93、Ascend950DT |
| 是否涉及量化 | 是/否，量化粒度 |

如果是已有算子，必须阅读：

- `op_host/`：OpDef、infer shape、tiling、aclnn 包装。
- `op_kernel/`：kernel 入口、tiling data、核心计算和数据流。
- 绑定层：PyTorch/pybind/torch extension 接口。
- 测试：UT、ST、benchmark 或现有精度对比脚本。
- 既有设计说明或 feature guide。

### Step 2: 逐节生成内容

按 `references/design-template.md` 的章节填充。

#### 第1节：需求

- 需求背景和描述：用 5W2H 组织；无法确认客户/时间时写"项目内 FLA/vLLM Ascend 场景"并注明待业务确认。
- 精度标准：说明标杆算子、通过标准、验收范围。
- 性能标准：说明性能基线、目标、验收 shape。
- 业务 shape：列出已有测试/benchmark 中的 shape，并补充泛化范围。

#### 第2节：约束和周边影响评估

- 周边依赖：docker、CANN、PyTorch、torch_npu、构建脚本或 CI 约束。
- 支持芯片型号：从 OpDef、CMake、构建脚本或文档中提取。

#### 第3节：算子功能及定义

- 功能分析：写清公式、流程、量化/非量化说明。
- 参数说明：使用模板参数表，字段至少包含参数名、描述、必选性、dtype、format、shape、值域、连续性、对齐、空 tensor、nan/inf。
- 其他支持：图模式、确定性、反向测试、原型定义、接口定义。
- aclnn 接口：列出 `aclnn{OpName}GetWorkspaceSize`、`aclnn{OpName}` 和 pybind/torch binding 签名。

#### 第4节：详细方案设计

- 计算逻辑：说明主流程、核间/核内循环、流水和同步。
- Tiling 方案：多核切分、UB 切分、workspace/buffer 规划、分支场景、`TilingData` 字段。
- Kernel 方案：TilingKey、模板条件、Ascend C API、GM/UB/workspace 数据流、内存管理。

#### 第5节：测试策略

- 测试参数：业务 shape、泛化 shape、输入值域和异常值域。
- 测试脚本：CPU golden、NPU 小算子或既有标杆实现。
- 标注需要真实 NPU 才能验证的测试。

### Step 3: 输出文档

- 将设计文档写入 `{operator_name}/docs/DESIGN.md`。
- 如需要，将 aclnn 接口文档写入 `{operator_name}/docs/aclnn{OperatorName}.md`。
- 保持 Markdown 表格和代码块可读，避免超长无结构段落。

## 完成检查

- [ ] 文档章节完整，覆盖需求、约束、功能、详细方案和测试策略。
- [ ] 所有接口签名与源码一致。
- [ ] shape/dtype/attr/tiling/workspace 字段可追溯到源码或用户输入。
- [ ] 无法确认的信息已明确标注，未编造链接或性能数字。
- [ ] 若生成仓库文件，已运行必要的格式/Markdown 基础检查。
