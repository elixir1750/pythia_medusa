# Pythia Medusa

一个基于 `EleutherAI/pythia-70m-deduped` 的分阶段 Medusa-1 实现仓库。

这个仓库是按照 `docs/pythia_medusa_plan/` 中的阶段文档逐步实现的，目标是先做一个**正确、清晰、可复现**的 Medusa 原型，再逐步逼近更完整的加速实现。

如果你现在更关心“怎么训练、怎么跑、怎么评估”，可以直接看：

- [docs/training_and_usage_guide.md](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/docs/training_and_usage_guide.md:1)

当前仓库已经覆盖：

- Phase 1：Baseline generation / LM evaluation / prompt sets / 结果导出
- Phase 2：MedusaConfig / Medusa heads / Medusa wrapper / checkpoint 约定
- Phase 3：correctness-first Medusa generation loop
- Phase 4：冻结 base model、只训练 Medusa heads 的最小训练链路
- Phase 5：baseline vs Medusa 的统一评估与生成对比
- Phase 6：可重复的 generation benchmark

当前实现的定位是：

- 优先正确性和接口清晰度
- 优先可测试、可调试
- Phase 3 采用文档建议的 Path A
- 还没有做 GPTNeoX 专用 tree-attention 注入和真正的高性能加速路径

如果你把它理解成“Medusa 研究原型 + 课程项目仓库”，这个定位会比较准确。

## 仓库目标

这个仓库主要解决四件事：

1. 建立一个稳定的 Pythia baseline，作为后续对比参照
2. 在 Pythia 上叠加 Medusa 多头 future-token predictor
3. 用最小训练与评估链路验证这些 heads 是否学到有用信息
4. 用统一日志和 benchmark 观察 Medusa 是否带来速度或行为差异

## 当前实现状态

已完成：

- baseline 文本生成
- baseline teacher-forcing LM evaluation
- 内置共享 prompt 集
- MedusaConfig
- ResBlock / MedusaHead / MedusaHeadStack
- MedusaModel wrapper
- Medusa head checkpoint save/load
- correctness-first Medusa generation loop
- Medusa training dataset / collator / loss / trainer
- baseline vs Medusa evaluation
- generation benchmark
- YAML 配置文件
- shell 脚本入口
- 对应 smoke tests / pipeline tests

尚未追求的能力：

- GPTNeoX 专用 tree verification 优化
- Medusa-aware KV cache 注入
- 真正意义上的高性能单步树验证
- 论文级别的吞吐优化

## 实现思路

### Phase 3 为什么是“correctness-first”

文档在 Phase 3 里明确建议优先实现 Path A：

- 不先写复杂的 GPTNeoX tree attention
- 先把 candidate generation、verification、acceptance、统计打通
- 让 generation loop 可解释、可测试、可复现

现在仓库里的 `medusa_generate.py` 默认已经切到 `tree_verify` 路径：Medusa heads 先给出一条线性的 candidate future-token 链，再由 base model 用一次 verification forward 验证整条链；同时保留 `serial` 作为调试后备。

所以当前实现不是“论文里最完整、最快”的版本，而是：

- 更容易验证正确性
- 更容易写单元测试
- 更容易在 Phase 4/5/6 中复用
- 更接近“Medusa candidate + 一次 attention 验证”的论文思路

### 当前 Medusa decode 的简化逻辑

一次生成轮次大致是：

1. 用 `MedusaModel` 对当前序列前向
2. 取基础 logits 的 next token 作为 fallback
3. 取 Medusa 多头 logits 作为候选 future tokens
4. 把这条 candidate 链组织成线性 tree，并构造 tree verification mask
5. 用基础模型对这条 tree 做一次 verification forward
6. 计算 accepted prefix length
7. 如果一个 candidate 都没接收，就退回基础模型 token，保证生成一定前进

这意味着：

- 逻辑正确性强
- acceptance 统计是可解释的
- 比早期串行逐 token 验证更接近 Medusa 原意
- 但还没有做到多分支 tree + KV cache 的完整高性能版本

## 项目结构

```text
pythia_medusa/
├── docs/pythia_medusa_plan/
├── configs/
│   ├── eval/
│   │   ├── benchmark_gen.yaml
│   │   └── eval_lm.yaml
│   └── train/
│       └── medusa_train_small.yaml
├── scripts/
│   ├── benchmark_compare.sh
│   └── train_medusa1.sh
├── src/pythia_medusa/
│   ├── data/
│   │   └── prompt_sets.py
│   ├── eval/
│   │   ├── benchmark_generation.py
│   │   ├── compare_models.py
│   │   ├── eval_heads.py
│   │   ├── eval_language_modeling.py
│   │   └── metrics.py
│   ├── generation/
│   │   ├── base_generate.py
│   │   ├── candidate_utils.py
│   │   ├── medusa_generate.py
│   │   ├── posterior_utils.py
│   │   └── tree_utils.py
│   ├── models/
│   │   ├── medusa_config.py
│   │   ├── medusa_heads.py
│   │   └── medusa_model.py
│   ├── training/
│   │   ├── collator.py
│   │   ├── dataset.py
│   │   ├── losses.py
│   │   └── trainer.py
│   └── utils/
│       ├── io.py
│       └── profiling.py
└── tests/
```

## 环境准备

建议使用 Python `3.10` 或 `3.11`。

原因很简单：

- `torch` 在这些版本上通常最稳定
- Hugging Face + PyTorch 的组合在 `3.10 / 3.11` 上更省心
- 如果你现在本机是更高版本 Python，建议直接新建一个独立环境

### 方案 1：conda

```bash
conda create -n pythia-medusa python=3.11 -y
conda activate pythia-medusa
pip install -r requirements.txt
```

如果你更想用 `3.10`：

```bash
conda create -n pythia-medusa python=3.10 -y
conda activate pythia-medusa
pip install -r requirements.txt
```

### 方案 2：venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 依赖

`requirements.txt` 当前包含：

```text
torch>=2.2
transformers>=4.40
pytest>=8.0
PyYAML>=6.0
```

说明：

- `torch`：模型前向、训练、loss、benchmark
- `transformers`：加载 Pythia tokenizer 和 causal LM
- `pytest`：本地测试
- `PyYAML`：读取训练/评估/benchmark 配置文件

## 模型是从哪里来的

默认模型名是：

```text
EleutherAI/pythia-70m-deduped
```

仓库不会把这个模型权重直接放在项目里。当前实现使用 Hugging Face 的标准加载方式：

- `AutoTokenizer.from_pretrained(...)`
- `AutoModelForCausalLM.from_pretrained(...)`

所以运行时行为是：

- 本地缓存里已有模型时，直接读取缓存
- 本地没有时，首次运行会尝试联网下载

你也可以把 `--model` 直接写成本地目录：

```bash
python -m pythia_medusa.generation.base_generate \
  --model /path/to/local/pythia-70m-deduped \
  --prompt "The capital of France is"
```

## 数据格式

### 文本生成 prompt 文件

支持 `jsonl`，每行一条记录。可识别字段：

- `prompt`
- `text`
- `input`

示例：

```jsonl
{"prompt": "The capital of France is", "name": "p1", "bucket": "short_factual"}
{"text": "Once upon a time, in a city powered by wind,", "name": "p2", "bucket": "short_continuation"}
{"input": "Write three bullet points about recycling.", "name": "p3", "bucket": "medium_instruction"}
```

### 训练/评估数据

支持两种最小格式：

1. `.jsonl`
2. `.txt`

`jsonl` 默认读取 `text` 字段：

```jsonl
{"text": "Paris is the capital of France."}
{"text": "Water freezes at zero degrees Celsius."}
```

纯文本文件则默认一行一条：

```text
Paris is the capital of France.
Water freezes at zero degrees Celsius.
```

如果你不想手工整理数据，现在仓库已经带了一个下载并导出训练数据的入口，能直接把 Hugging Face 自然文本数据集转换成当前 trainer 可读的 `jsonl`：

```bash
python -m pythia_medusa.data.prepare_text_dataset \
  --preset wikitext-2 \
  --train-output data/train.jsonl \
  --valid-output data/valid.jsonl \
  --manifest-output data/dataset_manifest.json \
  --chunker tokenizer \
  --tokenizer EleutherAI/pythia-70m-deduped \
  --chunk-size 128
```

内置支持：

- `wikitext-2`
- `wikitext-103`
- `pg19`

如果你的作业想优先贴近 Pythia 的常见自然文本分布，建议先用 `wikitext-2` 或 `wikitext-103` 做 stage1，再把 self-distill 作为对照实验补上。

## 内置 prompt 集

在 `src/pythia_medusa/data/prompt_sets.py` 中提供了共享 prompt 集：

- `short_factual`
- `short_continuation`
- `medium_reasoning`
- `medium_instruction`
- `all`

这套 prompt 集会被 baseline generation、Medusa generation、模型对比和 benchmark 共同复用。

## Phase 1：Baseline 使用方式

### baseline 文本生成

单条 prompt：

```bash
python -m pythia_medusa.generation.base_generate \
  --model EleutherAI/pythia-70m-deduped \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0.8
```

贪心生成：

```bash
python -m pythia_medusa.generation.base_generate \
  --model EleutherAI/pythia-70m-deduped \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --greedy
```

使用内置 prompt 集：

```bash
python -m pythia_medusa.generation.base_generate \
  --prompt-set short_factual \
  --limit 3 \
  --output outputs/baseline_generations.json
```

从 `jsonl` 文件读取：

```bash
python -m pythia_medusa.generation.base_generate \
  --prompt-file data/prompts.jsonl \
  --output outputs/baseline_generations.jsonl
```

### baseline LM evaluation

```bash
python -m pythia_medusa.eval.eval_language_modeling \
  --model EleutherAI/pythia-70m-deduped \
  --dataset data/valid.jsonl \
  --output outputs/baseline_eval.json \
  --csv-output outputs/baseline_eval.csv
```

输出指标包括：

- `loss`
- `perplexity`
- `num_tokens`
- `num_examples`
- `skipped_examples`

## Phase 2：Medusa 结构

### 核心对象

Phase 2 的三个关键模块：

- `MedusaConfig`
- `MedusaHeadStack`
- `MedusaModel`

### 最小加载示例

```python
from pythia_medusa.models import MedusaModel

model = MedusaModel.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    medusa_num_heads=3,
    medusa_num_layers=1,
)
```

### 前向输出示例

```python
outputs = model(
    **tokenizer("The capital of France is", return_tensors="pt"),
    medusa_forward=True,
    output_orig=True,
)

print(outputs.logits.shape)
print(outputs.medusa_logits.shape)
```

返回值语义：

- `outputs.logits`：基础模型原始 logits，shape `[batch, seq, vocab]`
- `outputs.medusa_logits`：Medusa logits，shape `[num_heads, batch, seq, vocab]`
- `outputs.hidden_states`：最后一层 hidden states
- `outputs.base_outputs`：底层 base model 原始输出

### checkpoint 格式

调用：

```python
model.save_medusa_checkpoint("outputs/medusa_ckpt")
```

会保存：

- `medusa_config.json`
- `medusa_heads.pt`
- `medusa_metadata.json`

重新加载：

```python
from pythia_medusa.models import MedusaModel

model = MedusaModel.from_medusa_checkpoint("outputs/medusa_ckpt")
```

## Phase 3：Medusa generation

### 设计说明

当前实现的是简化版、正确性优先的 decoding loop。

它做了这些事：

- 从 base logits 和 Medusa logits 生成候选 future tokens
- 默认用 `tree_verify` 对线性 candidate 链做单次 verification forward
- 计算 accepted prefix length
- 记录每轮 acceptance 统计
- 保证在 candidate 全部拒绝时仍然至少前进一步

它没有做这些事：

- 多分支 top-k tree 展开
- Medusa-aware KV cache 优化
- 完整的 GPTNeoX 高性能 tree decode

### CLI 示例

```bash
python -m pythia_medusa.generation.medusa_generate \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_small/checkpoint-epoch01-step00001 \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --greedy \
  --output outputs/medusa_generation.json
```

也可以直接用未训练 checkpoint 的随机 Medusa heads：

```bash
python -m pythia_medusa.generation.medusa_generate \
  --model EleutherAI/pythia-70m-deduped \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --greedy
```

说明：

- 能跑
- 能返回 acceptance 统计
- 默认使用 `tree_verify`
- 输出质量不会有训练后那么稳定

如果你想回退到更容易调试的旧路径，也可以显式加：

```bash
--verify-mode serial
```

### 输出字段

Medusa generation 会输出：

- `prompt`
- `generated_text`
- `full_text`
- `prompt_tokens`
- `generated_tokens`
- `latency_sec`
- `tokens_per_sec`
- `rounds`
- `accept_lengths`
- `average_accept_length`
- `accept_length_histogram`
- `round_traces`

### tree mask 语义

树语义在 `src/pythia_medusa/generation/tree_utils.py` 中有明确文档化：

- root 节点表示当前已验证前缀
- 每个节点可见自己
- 每个节点可见 root
- 每个节点可见祖先
- sibling / 无关分支互相不可见

当前 Phase 3 用的是线性树路径，但辅助函数已经把祖先关系和 mask 语义独立出来了，方便以后升级成更完整的 GPTNeoX tree 版本。

## Phase 4：训练 Medusa heads

### 训练目标

Phase 4 的最小训练目标是：

- head 0 预测 `+1`
- head 1 预测 `+2`
- head 2 预测 `+3`

训练策略：

- 冻结 base model 所有参数
- 只训练 `medusa_heads`

### 训练配置文件

默认示例配置：

- [configs/train/medusa_train_small.yaml](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/configs/train/medusa_train_small.yaml:1)

### Python CLI

```bash
python -m pythia_medusa.training.trainer \
  --config configs/train/medusa_train_small.yaml
```

或者直接传参数：

```bash
python -m pythia_medusa.training.trainer \
  --model EleutherAI/pythia-70m-deduped \
  --dataset data/train.jsonl \
  --valid-dataset data/valid.jsonl \
  --output-dir outputs/medusa_train_small \
  --seq-len 128 \
  --batch-size 8 \
  --lr 1e-3 \
  --num-epochs 1 \
  --medusa-num-heads 3 \
  --medusa-num-layers 1
```

当前 trainer 默认会在加载 `jsonl` 后先做一次预分词，再交给 DataLoader，适合课程作业里频繁小步试跑；如果你只想快速验证训练链路，也可以加 `--max-steps` 提前停止：

```bash
python -m pythia_medusa.training.trainer \
  --model EleutherAI/pythia-70m-deduped \
  --dataset data/train.jsonl \
  --valid-dataset data/valid.jsonl \
  --output-dir outputs/medusa_train_smoke \
  --seq-len 64 \
  --batch-size 2 \
  --lr 1e-5 \
  --num-epochs 1 \
  --max-steps 8 \
  --medusa-num-heads 3 \
  --medusa-num-layers 1
```

### 当前针对 Pythia-70M 的改进

围绕 `EleutherAI/pythia-70m-deduped`，当前仓库已经补齐了几项比较关键的工程化改进：

- 训练目标保持不变：冻结 backbone，只训练额外的 `medusa_heads`
- Medusa head 现在只学习 hidden-state refinement，最终统一复用 base model 的 `lm_head` 投到词表，而不是每个 head 再单独训练一套大的 `hidden -> vocab` 投影
- 训练数据支持先预分词再进入 DataLoader，减少了“每个 batch 现场 tokenizer”带来的额外开销
- trainer 增加了 `tqdm` 进度条、`--max-steps` 和 `--no-pretokenize-dataset`，更适合做 smoke test 和小规模课程实验
- Medusa loss 的聚合改成了更稳定的 `float32` 路径，避免在 `fp16/MPS` 下因为大张量求和而直接出现 `NaN`
- 如果训练中出现 non-finite loss，会在 `backward()` 前直接中止，避免把参数继续污染
- 在 Apple Silicon 上，如果指定 `--device mps`，trainer 会自动采用“base model 在 `mps`，medusa heads 在 `cpu float32`”的分离训练方式，绕开 `MPS + optimizer.step()` 对小头部训练不稳定的问题
- 数据准备链路已经补齐，可以直接从 `WikiText` / `PG-19` 下载自然文本并导出成当前 trainer 可读的 `jsonl`

这些改进的目标不是改变 Pythia-70M 的 backbone，而是让它更适合作为课程作业里的 Medusa 实验底座：

- 更容易稳定训练
- 更容易快速试跑
- 更方便比较自然文本训练和 self-distill 对照实验
- 更方便后续继续推进 tree-mask / verification 这条推理加速主线

### shell 脚本

```bash
bash scripts/train_medusa1.sh
```

或者指定配置文件：

```bash
bash scripts/train_medusa1.sh configs/train/medusa_train_small.yaml
```

### 训练输出

训练结束后，输出目录中通常会有：

- `training_summary.json`
- `run_summary.json`
- `train_log.csv`
- `checkpoint-epochXX-stepXXXXX/`

checkpoint 目录会包含：

- `medusa_config.json`
- `medusa_heads.pt`
- `medusa_metadata.json`
- `optimizer.pt`

如果启用 `save_tokenizer`，还会把 tokenizer 一起写入 checkpoint 目录。

### 训练中记录的指标

- 总 loss
- 每个 head 的 loss
- 每个 head 的 accuracy
- 每个 head 的 valid token 数

## Phase 5：评估与对比

### 静态 LM evaluation

仓库会对同一份数据评估：

- baseline loss / perplexity
- medusa wrapper 走 baseline path 时的 loss / perplexity

这个对比的目的不是看 Medusa 是否更快，而是确认：

- 包装器没有破坏基础模型推理行为

### head-wise evaluation

仓库还会统计：

- `head1_acc`
- `head2_acc`
- `head3_acc`
- 对应的 per-head loss

### generation comparison

Phase 5 会在固定 prompt 集上导出：

- baseline generated text
- medusa generated text
- output token 数
- latency
- average accept length

### compare CLI

```bash
python -m pythia_medusa.eval.compare_models \
  --config configs/eval/eval_lm.yaml
```

或者直接传参数：

```bash
python -m pythia_medusa.eval.compare_models \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_small/checkpoint-epoch01-step00001 \
  --dataset data/valid.jsonl \
  --output-dir outputs/eval_compare \
  --prompt-set all \
  --max-new-tokens 32
```

### compare 输出文件

默认会写出：

- `comparison_summary.json`
- `comparison_summary.csv`
- `generation_compare.jsonl`

## Phase 6：benchmark

### benchmark 目标

这个阶段的目标是：

- 测量
- 可复现
- 统一记录

不是“证明 Medusa 一定更快”。

对于这个仓库当前的 correctness-first 实现，出现以下情况都是合理的：

- Medusa 稍微快一点
- Medusa 根本没更快
- Medusa 更适合先做研究日志，而不是直接做速度结论

### benchmark CLI

```bash
python -m pythia_medusa.eval.benchmark_generation \
  --config configs/eval/benchmark_gen.yaml
```

或者直接传参数：

```bash
python -m pythia_medusa.eval.benchmark_generation \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_small/checkpoint-epoch01-step00001 \
  --output-dir outputs/benchmark_compare \
  --prompt-set all \
  --max-new-tokens 32 \
  --repeat 5 \
  --warmup 1
```

### shell 脚本

```bash
bash scripts/benchmark_compare.sh
```

脚本支持通过环境变量覆盖：

- `MODEL_NAME`
- `CHECKPOINT_PATH`
- `OUTPUT_DIR`
- `PROMPT_SET`
- `MAX_NEW_TOKENS`
- `REPEAT_COUNT`
- `WARMUP_COUNT`

### benchmark 输出

会生成：

- `benchmark_summary.json`
- `benchmark_summary.csv`
- `baseline_runs.jsonl`
- `medusa_runs.jsonl`

典型 summary 字段包括：

- `avg_latency_sec`
- `avg_tokens_per_sec`
- `avg_accept_length`
- `latency_stats`
- `throughput_stats`

## 一套推荐工作流

如果你想完整跑一遍，推荐顺序是：

1. 安装环境
2. 跑 baseline generation
3. 跑 baseline LM evaluation
4. 准备一份小训练集，训练 Medusa heads
5. 用 compare_models 对比 baseline vs Medusa
6. 最后跑 benchmark

一个最小示例：

```bash
conda create -n pythia-medusa python=3.11 -y
conda activate pythia-medusa
pip install -r requirements.txt

python -m pythia_medusa.generation.base_generate \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --output outputs/demo_baseline.json
```

训练：

```bash
python -m pythia_medusa.training.trainer \
  --config configs/train/medusa_train_small.yaml
```

评估：

```bash
python -m pythia_medusa.eval.compare_models \
  --config configs/eval/eval_lm.yaml
```

benchmark：

```bash
python -m pythia_medusa.eval.benchmark_generation \
  --config configs/eval/benchmark_gen.yaml
```

## 代码入口索引

如果你想按功能读代码，推荐从这些入口开始：

### 基线

- [src/pythia_medusa/generation/base_generate.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/generation/base_generate.py:53)
- [src/pythia_medusa/eval/eval_language_modeling.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/eval/eval_language_modeling.py:24)

### Medusa 模型结构

- [src/pythia_medusa/models/medusa_config.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/models/medusa_config.py:13)
- [src/pythia_medusa/models/medusa_heads.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/models/medusa_heads.py:9)
- [src/pythia_medusa/models/medusa_model.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/models/medusa_model.py:33)

### Medusa 生成

- [src/pythia_medusa/generation/candidate_utils.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/generation/candidate_utils.py:8)
- [src/pythia_medusa/generation/posterior_utils.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/generation/posterior_utils.py:7)
- [src/pythia_medusa/generation/tree_utils.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/generation/tree_utils.py:7)
- [src/pythia_medusa/generation/medusa_generate.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/generation/medusa_generate.py:129)

### 训练

- [src/pythia_medusa/training/dataset.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/training/dataset.py:10)
- [src/pythia_medusa/training/collator.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/training/collator.py:8)
- [src/pythia_medusa/training/losses.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/training/losses.py:10)
- [src/pythia_medusa/training/trainer.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/training/trainer.py:97)

### 评估与 benchmark

- [src/pythia_medusa/eval/eval_heads.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/eval/eval_heads.py:17)
- [src/pythia_medusa/eval/compare_models.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/eval/compare_models.py:72)
- [src/pythia_medusa/eval/benchmark_generation.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/eval/benchmark_generation.py:65)
- [src/pythia_medusa/utils/profiling.py](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/src/pythia_medusa/utils/profiling.py:9)

## 测试

当前测试包括：

- baseline generation / evaluation smoke test
- Medusa shape test
- Medusa wrapper forward test
- Medusa generation loop test
- training pipeline test
- evaluation / compare / benchmark test

运行全部测试：

```bash
pytest
```

只跑关键测试：

```bash
pytest tests/test_generation_smoke.py
pytest tests/test_medusa_forward.py
pytest tests/test_medusa_generation.py
pytest tests/test_training_pipeline.py
pytest tests/test_eval_pipeline.py
```

## 常见问题

### 1. `No module named torch`

没有安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 首次加载模型很慢

因为 `transformers` 可能正在从 Hugging Face 下载：

```text
EleutherAI/pythia-70m-deduped
```

首次慢是正常的，后续通常会走本地缓存。

### 3. Medusa 没有更快，是不是实现错了

不一定。

当前仓库的重点是：

- correctness-first
- 可测可比
- 有 acceptance / latency / throughput 统计

它不是一个已经针对 GPTNeoX 深度优化过的生产级 Medusa 推理库。

### 4. 为什么 benchmark 和 generation 都分了 baseline / medusa 两套

因为 Phase 5 和 Phase 6 的目标不同：

- Phase 5 更关心行为和质量对比
- Phase 6 更关心 timing / throughput

所以代码虽然复用底层模块，但输出和汇总的关注点会不一样。

## 后续可以继续做什么

如果你想继续把这个仓库往“更像论文里的 Medusa”推进，下一步通常会是：

1. 给 GPTNeoX 增加更完整的 tree verification 支持
2. 引入 Medusa-aware cache 和 attention mask 注入
3. 把多轮逐步验证改成更接近单轮树验证
4. 再重新测 benchmark，看吞吐是否真正变好

如果你想把它往“课程项目展示”方向推进，下一步更建议：

1. 准备一份稳定的小数据集
2. 训一个短 checkpoint
3. 跑 compare 和 benchmark
4. 固定一组图表和结果表

## 对照文档

规划文档在：

- [docs/pythia_medusa_plan/README.md](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/docs/pythia_medusa_plan/README.md:1)
- [docs/pythia_medusa_plan/phase_01_baseline.md](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/docs/pythia_medusa_plan/phase_01_baseline.md:1)
- [docs/pythia_medusa_plan/phase_02_medusa_structure.md](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/docs/pythia_medusa_plan/phase_02_medusa_structure.md:1)
- [docs/pythia_medusa_plan/phase_03_medusa_generation.md](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/docs/pythia_medusa_plan/phase_03_medusa_generation.md:1)
- [docs/pythia_medusa_plan/phase_04_training.md](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/docs/pythia_medusa_plan/phase_04_training.md:1)
- [docs/pythia_medusa_plan/phase_05_evaluation.md](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/docs/pythia_medusa_plan/phase_05_evaluation.md:1)
- [docs/pythia_medusa_plan/phase_06_speed_benchmark.md](/Users/ken_elixir/Desktop/Offline%20Courses/NLP形式逻辑语音/NLP/大作业/pythia_medusa/docs/pythia_medusa_plan/phase_06_speed_benchmark.md:1)
