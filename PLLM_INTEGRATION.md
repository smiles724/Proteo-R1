● 非常好的问题！基于你的情况，我给出明确的建议：

  🎯 推荐方案：选择路径 1（模仿 Qwen2.5-VL）

  核心理由

  1. 时间成本对比

  | 维度           | 路径1: Qwen2.5-VL 风格    | 路径2: BAGEL 风格           |
  |--------------|-----------------------|-------------------------|
  | Processor 实现 | 🟢 简单（Template-based） | 🔴 复杂（Packed sequence）  |
  | Dataset 实现   | 🟢 标准（Map/Iterable）   | 🟡 中等（需理解 packing）      |
  | Collator 实现  | 🟢 简单（Padding）        | 🔴 复杂（Index offsetting） |
  | 调试难度         | 🟢 低（标准流程）            | 🔴 高（嵌套 mask）           |
  | 预计时间         | 1-2 周                 | 3-4 周                   |

  2. 稳定性对比

  路径1（Qwen2.5-VL）:
  - ✅ 已有完整的参考实现
  - ✅ 文档齐全，社区成熟
  - ✅ 不依赖复杂的 packing 机制
  - ✅ 即使未来不做生成，也是完整的解决方案

  路径2（BAGEL）:
  - ⚠️ 必须理解 packed sequence（学习成本高）
  - ⚠️ 必须理解 nested attention masks（容易出错）
  - ⚠️ 如果队友未来不做生成端，前期投入的 packing 复杂度没有价值

  3. 团队协作视角

  路径1 的优势：
  # 你的实现（理解端）
  class PLLMForProteinUnderstanding(PreTrainedModel):
      """标准的 VLM 架构，易于理解"""
      def forward(self, input_ids, protein_input_ids, structure_input_ids, ...):
          # 清晰的数据流
          seq_features = self.protein_encoder(protein_input_ids)
          struct_features = self.structure_encoder(structure_input_ids)
          # ...
          return {"loss": ce_loss, "logits": logits}

  # 队友未来添加生成端（如果需要）
  class ProteinBagel(PreTrainedModel):
      """复杂的理解+生成架构"""
      def forward(self, sequence_length, packed_text_ids, ...):
          # 可以参考你的 PLLMForProteinUnderstanding
          # 然后扩展 packed sequence 逻辑

  好处：
  - ✅ 你的代码独立可用，不依赖生成端
  - ✅ 队友可以基于你的清晰实现，再决定是否需要 BAGEL 复杂度
  - ✅ 如果最终不做生成，路径1 不浪费时间

  路径2 的风险：
  - ❌ 你必须预先实现 packed sequence（即使可能用不上）
  - ❌ 队友必须完全理解你的 packing 逻辑才能添加生成端
  - ❌ 如果最终不做生成，前期 30-40% 的工作白费

  ---
  📋 具体实施方案：路径1 详细步骤

  Phase 1: 核心模型实现（Week 1）

  1.1 创建模型配置

  # src/lmms_engine/models/pllm/configuration_pllm.py

  from transformers import PretrainedConfig, Qwen2Config

  class PLLMConfig(PretrainedConfig):
      model_type = "pllm"

      def __init__(
          self,
          # LLM 配置
          llm_config: dict = None,

          # 编码器配置
          protein_encoder_path: str = "facebook/esm2_t33_650M_UR50D",
          structure_encoder_path: str = "facebook/esm_foldseek_t30_150M",

          # 投影器配置
          joint_projector: bool = True,
          proj_hid: int = 1024,
          dropout: float = 0.1,

          # 特殊 token
          seq_token: str = "<aa_seq>",
          struct_token: str = "<3d_struct>",

          **kwargs,
      ):
          super().__init__(**kwargs)
          self.llm_config = Qwen2Config(**llm_config) if llm_config else None
          # ...

  1.2 创建模型主体

  # src/lmms_engine/models/pllm/modeling_pllm.py

  class PLLMForProteinUnderstanding(PreTrainedModel):
      """
      蛋白质理解模型（纯 VLM 架构，不涉及生成）

      架构: protein_encoder + structure_encoder + prefix_mlp + Qwen2
      """
      config_class = PLLMConfig

      def __init__(self, config: PLLMConfig):
          super().__init__(config)

          # 1. LLM 后端
          self.language_model = Qwen2ForCausalLM(config.llm_config)

          # 2. 编码器（直接复用你的实现）
          self.protein_encoder = ProteinEncoder(...)
          self.structure_encoder = StructureEncoder(...)

          # 3. 投影器（直接复用）
          self.prefix_mlp = PrefixProjector(...)

      def forward(
          self,
          input_ids: torch.LongTensor,
          attention_mask: torch.Tensor,
          labels: Optional[torch.LongTensor] = None,

          # 蛋白质输入
          protein_input_ids: Optional[torch.LongTensor] = None,
          protein_attention_mask: Optional[torch.Tensor] = None,
          structure_input_ids: Optional[torch.LongTensor] = None,
          structure_attention_mask: Optional[torch.Tensor] = None,

          **kwargs,
      ):
          """
          完全模仿 Qwen2.5-VL 的接口
          """
          # 1. 文本 embedding
          inputs_embeds = self.language_model.model.embed_tokens(input_ids)

          # 2. 编码蛋白质
          if protein_input_ids is not None:
              seq_tok, seq_mask, _ = self.protein_encoder(
                  protein_input_ids, protein_attention_mask
              )
              struct_tok, struct_mask, _ = self.structure_encoder(
                  structure_input_ids, structure_attention_mask
              )

              # 3. 投影
              seq_prefix = self.prefix_mlp(seq_tok)
              struct_prefix = self.prefix_mlp(struct_tok)

              # 4. 替换特殊 token（与你的 PLLM 实现一致）
              seq_token_mask = input_ids == self.config.seq_token_id
              struct_token_mask = input_ids == self.config.struct_token_id

              inputs_embeds[seq_token_mask] = seq_prefix[seq_mask]
              inputs_embeds[struct_token_mask] = struct_prefix[struct_mask]

          # 5. LLM forward
          outputs = self.language_model(
              inputs_embeds=inputs_embeds,
              attention_mask=attention_mask,
          )

          # 6. 计算损失
          loss = None
          if labels is not None:
              logits = outputs.logits
              loss = F.cross_entropy(
                  logits.view(-1, logits.size(-1)),
                  labels.view(-1),
                  ignore_index=-100,
              )

          return {
              "loss": loss,
              "logits": outputs.logits,
          }

  关键点：
  - ✅ 接口与 Qwen2.5-VL 完全一致
  - ✅ 内部逻辑与你的 PLLM 完全一致
  - ✅ 队友看到这个代码会立即理解

  ---
  Phase 2: Processor 实现（Week 1）

  # src/lmms_engine/datasets/processor/pllm_processor.py

  from transformers import Qwen2Tokenizer, EsmTokenizer
  from lmms_engine.mapping_func import register_processor

  @register_processor("pllm")
  class PLLMDataProcessor:
      def __init__(self, config: ProcessorConfig):
          self.tokenizer = Qwen2Tokenizer.from_pretrained(config.processor_name)
          self.protein_tokenizer = EsmTokenizer.from_pretrained(
              config.extra_kwargs.get("protein_encoder_path")
          )
          self.structure_tokenizer = EsmTokenizer.from_pretrained(
              config.extra_kwargs.get("structure_encoder_path")
          )

          # 添加特殊 token
          self.tokenizer.add_tokens(["<aa_seq>", "<3d_struct>"])
          self.seq_token_id = self.tokenizer.convert_tokens_to_ids("<aa_seq>")
          self.struct_token_id = self.tokenizer.convert_tokens_to_ids("<3d_struct>")

      def process(
          self,
          proteins: List[Dict],  # [{"aa_seq": "...", "threeDi_seq": "..."}]
          hf_messages,
          **kwargs,
      ):
          """
          简单的 template-based 处理（模仿 Qwen2.5-VL）
          """
          # 1. 使用 Chat Template 渲染
          text = self.tokenizer.apply_chat_template(
              hf_messages,
              tokenize=False,
              add_generation_prompt=False,
          )
          # 输出: "<|im_start|>user\n<aa_seq><3d_struct>What is the function?..."

          # 2. 扩展 <aa_seq> 和 <3d_struct> 占位符
          for protein in proteins:
              aa_len = len(protein["aa_seq"])
              struct_len = len(protein["threeDi_seq"])

              # 替换为 N 个占位符（与你的 PLLM 实现一致）
              text = text.replace("<aa_seq>", "<aa_seq>" * aa_len, 1)
              text = text.replace("<3d_struct>", "<3d_struct>" * struct_len, 1)

          # 3. Tokenize
          text_tokens = self.tokenizer(text, return_tensors="pt")

          # 4. 编码蛋白质序列
          protein_tokens = self.protein_tokenizer(
              [p["aa_seq"] for p in proteins],
              return_tensors="pt",
              padding=True,
          )
          structure_tokens = self.structure_tokenizer(
              [p["threeDi_seq"] for p in proteins],
              return_tensors="pt",
              padding=True,
          )

          # 5. 生成 labels（mask 掉 user 输入）
          labels = self._create_labels(text_tokens.input_ids, hf_messages)

          return {
              "input_ids": text_tokens.input_ids,
              "attention_mask": text_tokens.attention_mask,
              "labels": labels,
              "protein_input_ids": protein_tokens.input_ids,
              "protein_attention_mask": protein_tokens.attention_mask,
              "structure_input_ids": structure_tokens.input_ids,
              "structure_attention_mask": structure_tokens.attention_mask,
          }

      def _create_labels(self, input_ids, hf_messages):
          """生成 SFT labels（只监督 assistant 回复）"""
          labels = input_ids.clone()

          # 找到所有 assistant 回复的起始位置
          # （模仿 Qwen2.5-VL 的 label masking）
          # ...

          return labels

  关键优势：
  - ✅ 不需要 packed sequence
  - ✅ 不需要 nested attention masks
  - ✅ 标准的 padding + masking

  ---
  Phase 3: Dataset 实现（Week 1）

  # src/lmms_engine/datasets/naive/pllm_dataset.py

  @register_dataset("pllm")
  class PLLMDataset(MultiModalDataset):
      """蛋白质数据集（标准 Map-style）"""

      def load_from_json(self, data, data_folder=None):
          """
          从你的 PLLM 格式加载数据

          输入格式:
          {
            "pdb_id": "1ABC",
            "question": "What is the function?",
            "response": "This protein...",
            "chains": {
              "A": {"aa_seq": "MKTL...", "threeDi_seq": "dabc..."}
            }
          }
          """
          # 1. 解析蛋白质数据
          proteins = []
          for chain_id, chain_data in data["chains"].items():
              proteins.append({
                  "chain_id": chain_id,
                  "aa_seq": chain_data["aa_seq"],
                  "threeDi_seq": chain_data["threeDi_seq"],
              })

          # 2. 构建 messages（OpenAI 格式）
          messages = [
              {
                  "role": "user",
                  "content": [
                      {"type": "protein", "proteins": proteins},
                      {"type": "text", "text": data["question"]},
                  ],
              },
              {
                  "role": "assistant",
                  "content": [{"type": "text", "text": data["response"]}],
              },
          ]

          # 3. 转换为 HF 格式
          hf_messages = TrainUtilities.convert_open_to_hf(messages)

          # 4. 调用 processor
          inputs = self.processor.process(proteins=proteins, hf_messages=hf_messages)

          return inputs

      def get_collator(self):
          return VisionCollator(self.processor)  # 复用标准 Collator

  关键点：
  - ✅ 复用 MultiModalDataset 基类
  - ✅ 复用 VisionCollator（标准 padding）
  - ✅ 不需要自定义 Collator

  ---
  Phase 4: 配置文件（Week 1）

  # examples/pllm/example_config.yaml

  trainer_type: fsdp2_trainer

  dataset_config:
    dataset_type: pllm
    dataset_format: json
    processor_config:
      processor_name: Qwen/Qwen2.5-0.5B-Instruct
      processor_type: pllm
      extra_kwargs:
        protein_encoder_path: facebook/esm2_t33_650M_UR50D
        structure_encoder_path: facebook/esm_foldseek_t30_150M
    datasets:
      - path: /path/to/protein_data.json
        data_type: json
    packing: false  # 第一阶段不用 packing
    shuffle: true

  model_config:
    load_from_pretrained_path: null
    attn_implementation: flash_attention_2
    torch_dtype: bfloat16
    extra_kwargs:
      joint_projector: true
      proj_hid: 1024
      dropout: 0.1

  trainer_args:
    output_dir: ./output/pllm_training
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 4
    learning_rate: 1e-4
    max_steps: 1000
    bf16: true
    fsdp2: true
    fsdp_config:
      transformer_layer_cls_to_wrap: ["Qwen2DecoderLayer"]
    use_liger_kernel: true
    gradient_checkpointing: true

  ---
  🔄 未来扩展路径（如果需要生成端）

  队友的两个选择

  选择 A: 继续使用路径1风格（推荐）

  # 队友添加一个独立的生成模型
  class PLLMForProteinGeneration(PreTrainedModel):
      """蛋白质生成模型（独立于理解端）"""
      def __init__(self, config):
          # 可以参考你的理解端实现
          # 添加生成特有的组件（如结构解码器）
          ...

  # 或者创建一个联合模型
  class PLLMForUnifiedProtein(PreTrainedModel):
      """理解+生成统一模型（简单版）"""
      def __init__(self, config):
          # 复用你的理解端
          self.understanding_model = PLLMForProteinUnderstanding.from_pretrained(...)
          # 添加生成端
          self.generation_decoder = ProteinStructureDecoder(...)

      def forward(self, ...):
          if self.training_mode == "understanding":
              return self.understanding_model(...)
          else:
              # 生成逻辑
              ...

  优势：
  - ✅ 不需要重构你的理解端代码
  - ✅ 生成端可以独立开发和测试
  - ✅ 如果生成端失败，理解端不受影响

  选择 B: 迁移到 BAGEL 风格

  如果队友未来确实需要 BAGEL 的复杂度（例如需要统一的 attention mask），可以：

  # 队友创建 ProteinBagel
  class ProteinBagel(PreTrainedModel):
      """完整的 BAGEL 风格实现"""
      def __init__(self, config):
          # 1. 复用你的编码器和投影器（核心逻辑不变）
          self.protein_encoder = PLLMForProteinUnderstanding.protein_encoder
          self.structure_encoder = PLLMForProteinUnderstanding.structure_encoder
          self.prefix_mlp = PLLMForProteinUnderstanding.prefix_mlp

          # 2. 添加 packed sequence 逻辑
          # 3. 添加生成端组件
          ...

  关键点：
  - ✅ 你的核心模型组件（编码器、投影器）可以直接复用
  - ✅ 只需要重构数据流水线（Processor + Dataset）

  ---
  📊 两条路径的最终对比

  | 维度   | 路径1: Qwen2.5-VL 风格  | 路径2: BAGEL 风格 |
  |------|---------------------|---------------|
  | 实现时间 | ⭐⭐⭐⭐⭐ 1-2周          | ⭐⭐ 3-4周       |
  | 稳定性  | ⭐⭐⭐⭐⭐ 高（已验证）        | ⭐⭐⭐ 中（复杂）     |
  | 队友理解 | ⭐⭐⭐⭐⭐ 易懂            | ⭐⭐ 需学习        |
  | 独立性  | ⭐⭐⭐⭐⭐ 完全独立          | ⭐⭐⭐ 依赖生成端设计   |
  | 未来扩展 | ⭐⭐⭐⭐ 两个方向都可         | ⭐⭐⭐⭐⭐ 已优化     |
  | 代码复用 | ⭐⭐⭐⭐⭐ 85% 复用你的 PLLM | ⭐⭐⭐⭐ 70% 复用   |
  | 风险   | ⭐⭐⭐⭐⭐ 低             | ⭐⭐ 高（调试难）     |

  ---
  🎯 最终建议

  选择路径1，原因如下：

  1. 时间约束 → 路径1 快 50%
  2. 你对生成端不了解 → 路径1 不需要理解生成端
  3. 便于队友理解 → 路径1 使用标准 VLM 架构
  4. 独立可用 → 即使不做生成，路径1 也是完整方案
  5. 未来灵活 → 队友可基于路径1 扩展到 BAGEL（如果需要）

  具体行动计划（Week 1-2）

  Week 1:
  - Day 1-2: 创建 modeling_pllm.py（复用 80% 你的代码）
  - Day 3-4: 创建 pllm_processor.py（简单的 template-based）
  - Day 5: 创建 pllm_dataset.py + example_config.yaml

  Week 2:
  - Day 1-2: 单元测试（模型 forward、processor、dataset）
  - Day 3-4: 小规模训练测试（100 条数据）
  - Day 5: 文档 + 交接给队友

  成功标准

  ✅ 你的交付物：
  1. 可运行的 PLLM 理解端（在 lmms-engine 中）
  2. 清晰的代码和注释
  3. 简单的示例配置
  4. README 说明如何使用

  ✅ 队友可以：
  1. 直接运行你的理解端训练
  2. 理解你的架构（标准 VLM）
  3. 基于你的实现，独立开发生成端

  ---
  总结: 选择路径1（Qwen2.5-VL 风格）是稳妥、快速、风险最低的方案。如果未来确实需要 BAGEL 的复杂度，可以在路径1 的基础上渐进式迁移，而不是一开始就承担不必要的复杂度。