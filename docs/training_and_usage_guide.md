# Pythia Medusa 训练与使用指南

这份文档是仓库的实操手册，和 `docs/pythia_medusa_plan/` 里的阶段规划文档不同。

如果你现在的目标是：

- 把环境配起来
- 准备数据
- 训练一个 Medusa checkpoint
- 跑生成、评估和 benchmark

那优先看这份文档就够了。

## 1. 你会用到什么

这套仓库当前已经提供了：

- baseline 文本生成
- baseline 语言模型评估
- Medusa 结构封装
- Medusa generation
- Medusa heads 训练
- baseline vs Medusa 对比评估
- generation benchmark

核心入口：

- baseline generation：`python -m pythia_medusa.generation.base_generate`
- Medusa generation：`python -m pythia_medusa.generation.medusa_generate`
- 训练：`python -m pythia_medusa.training.trainer`
- 对比评估：`python -m pythia_medusa.eval.compare_models`
- benchmark：`python -m pythia_medusa.eval.benchmark_generation`

## 2. 环境准备

建议用 Python `3.10` 或 `3.11`。

### 2.1 conda 方案

```bash
conda create -n pythia-medusa python=3.11 -y
conda activate pythia-medusa
pip install -r requirements.txt
```

如果你更希望用 `3.10`：

```bash
conda create -n pythia-medusa python=3.10 -y
conda activate pythia-medusa
pip install -r requirements.txt
```

### 2.2 venv 方案

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.3 检查安装是否正常

```bash
python -m pythia_medusa.generation.base_generate --help
python -m pythia_medusa.training.trainer --help
python -m pythia_medusa.eval.compare_models --help
python -m pythia_medusa.eval.benchmark_generation --help
```

## 3. 模型来源

默认模型是：

```text
EleutherAI/pythia-70m-deduped
```

当前实现通过 Hugging Face 标准接口加载：

- `AutoTokenizer.from_pretrained(...)`
- `AutoModelForCausalLM.from_pretrained(...)`

所以：

- 本地有缓存时会直接用缓存
- 本地没有时首次运行会下载

如果你已经把模型下载到本地，也可以直接传本地目录：

```bash
python -m pythia_medusa.generation.base_generate \
  --model /path/to/local/pythia-70m-deduped \
  --prompt "The capital of France is"
```

## 4. 数据准备

### 4.1 训练/评估数据

最小支持两种格式：

- `.jsonl`
- `.txt`

`jsonl` 默认读 `text` 字段：

```jsonl
{"text": "Paris is the capital of France."}
{"text": "Water freezes at zero degrees Celsius."}
{"text": "The moon orbits the Earth."}
```

纯文本格式就是一行一条：

```text
Paris is the capital of France.
Water freezes at zero degrees Celsius.
The moon orbits the Earth.
```

推荐你先准备：

- `data/train.jsonl`
- `data/valid.jsonl`

如果你想直接下载更贴近 Pythia 预训练场景的自然文本，可以用仓库内置的数据准备脚本：

```bash
python -m pythia_medusa.data.prepare_text_dataset \
  --preset wikitext-2 \
  --train-output data/train.jsonl \
  --valid-output data/valid.jsonl \
  --manifest-output data/dataset_manifest.json \
  --chunker tokenizer \
  --tokenizer EleutherAI/pythia-70m-deduped \
  --chunk-size 128 \
  --max-train-records 2000 \
  --max-valid-records 400
```

内置 `preset`：

- `wikitext-2`：默认推荐，适合课程作业和快速验证
- `wikitext-103`：更大一些，适合主实验想做得更扎实时
- `pg19`：更长文本分布，适合做 continuation 风格补充实验

结合你现在的作业目标，建议：

- 主实验：`--preset wikitext-2` 或 `--preset wikitext-103`
- 对照实验：保留同一套评测脚本，再额外做一版 self-distill 数据

这个脚本会把 Hugging Face 数据集清洗并导出成当前 trainer 直接可读的 `jsonl`，不需要改训练命令。

### 4.2 prompt 数据

如果你想自己控制生成对比的 prompt，可以准备一个 `jsonl`：

```jsonl
{"prompt": "The capital of France is", "name": "p1", "bucket": "short_factual"}
{"prompt": "Once upon a time, in a city powered by wind,", "name": "p2", "bucket": "short_continuation"}
{"prompt": "Explain why the sum of two odd numbers is always even.", "name": "p3", "bucket": "medium_reasoning"}
```

仓库也自带 prompt 集，可以直接用：

- `short_factual`
- `short_continuation`
- `medium_reasoning`
- `medium_instruction`
- `all`

## 5. 先跑 baseline

建议先确认 baseline 是通的，再开始训 Medusa。

### 5.1 baseline 生成

```bash
python -m pythia_medusa.generation.base_generate \
  --model EleutherAI/pythia-70m-deduped \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --greedy \
  --output outputs/baseline_generation.json
```

### 5.2 baseline 评估

```bash
python -m pythia_medusa.eval.eval_language_modeling \
  --model EleutherAI/pythia-70m-deduped \
  --dataset data/valid.jsonl \
  --output outputs/baseline_eval.json \
  --csv-output outputs/baseline_eval.csv
```

如果 baseline 都跑不通，不建议直接进入训练。

## 6. 训练 Medusa heads

### 6.1 默认配置文件

仓库已经给了一个最小训练配置：

- `configs/train/medusa_train_small.yaml`

当前内容大意是：

- model: `EleutherAI/pythia-70m-deduped`
- seq_len: `128`
- batch_size: `8`
- lr: `1e-3`
- medusa_num_heads: `3`
- medusa_num_layers: `1`

### 6.2 直接用配置文件训练

```bash
python -m pythia_medusa.training.trainer \
  --config configs/train/medusa_train_small.yaml
```

当前 trainer 默认会在数据加载阶段先把文本预分词，再进入 DataLoader，这样比“每个 batch 现分词”更适合反复做课程实验。

### 6.3 用 shell 脚本训练

```bash
bash scripts/train_medusa1.sh
```

或者指定配置：

```bash
bash scripts/train_medusa1.sh configs/train/medusa_train_small.yaml
```

### 6.4 不用 YAML，直接命令行训练

```bash
python -m pythia_medusa.training.trainer \
  --model EleutherAI/pythia-70m-deduped \
  --dataset data/train.jsonl \
  --valid-dataset data/valid.jsonl \
  --output-dir outputs/medusa_train_small \
  --seq-len 128 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr 1e-3 \
  --num-epochs 1 \
  --medusa-num-heads 3 \
  --medusa-num-layers 1
```

如果你只是想快速试跑，确认训练不会报错、不会立刻出现 `NaN`，推荐先加一个 `--max-steps`：

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

### 6.5 当前针对 Pythia-70M 做了哪些改进

为了把 `EleutherAI/pythia-70m-deduped` 变成一个更适合课程作业实验的 Medusa 底座，当前实现已经做了这些改进：

- 保持 backbone 不变：冻结 base model，只训练新增的 `medusa_heads`
- 重新收紧了 Medusa head 的参数化方式：head 只学习 hidden-state refinement，最后统一复用 base model 的 `lm_head` 投到词表，而不是每个 head 自己再学一个巨大的词表投影
- 训练数据默认会先预分词，再进入 DataLoader，减少了 batch 内重复 tokenizer 的 Python 开销
- trainer 增加了 `tqdm` 进度条，训练时不再像“卡住了一样没有输出”
- trainer 增加了 `--max-steps`，方便先做 8-step 这种 smoke test，再决定是否扩大实验规模
- Medusa loss 聚合改成了更稳定的 `float32` 路径，避免 `fp16 / MPS` 场景下因为大张量求和先溢出而把总 loss 变成 `NaN`
- 如果 loss 变成 non-finite，trainer 会在 `backward()` 前直接报错并停止，避免继续污染参数
- 在 Apple Silicon 上，如果你指定 `--device mps`，trainer 会自动走“base model 在 `mps`，medusa heads 在 `cpu float32`”的分离训练，绕开 `MPS + optimizer.step()` 对小头部训练不稳定的问题
- 新增了自然文本数据准备 CLI，可以直接下载 `wikitext-2`、`wikitext-103`、`pg19` 并导出成当前训练入口可读的 `jsonl`

如果你这份大作业的目标是“在不改变 Pythia backbone 的前提下，尝试用 Medusa 做推理加速”，那这些改进主要解决的是：

- 训练能不能稳定跑起来
- 小规模实验能不能快速迭代
- 自然文本版和 self-distill 版能不能用统一入口比较

### 6.6 当前训练逻辑做了什么

训练时会：

- 加载基础 Pythia 模型
- 构建 Medusa heads
- 冻结 base model 参数
- 只让 `medusa_heads` 参与训练
- 计算每个 head 对未来 token 的交叉熵
- 记录 per-head loss 和 accuracy
- 每个 epoch 保存 checkpoint

### 6.7 训练输出目录会有什么

典型输出：

- `outputs/medusa_train_small/training_summary.json`
- `outputs/medusa_train_small/run_summary.json`
- `outputs/medusa_train_small/train_log.csv`
- `outputs/medusa_train_small/checkpoint-epoch01-step000xx/`

checkpoint 目录里通常包含：

- `medusa_config.json`
- `medusa_heads.pt`
- `medusa_metadata.json`
- `optimizer.pt`

如果启用了 tokenizer 保存，还会包含 tokenizer 文件。

## 7. 使用训练后的 checkpoint

训练完成后，先记住你的 checkpoint 路径。例如：

```text
outputs/medusa_train_small/checkpoint-epoch01-step00010
```

后续的 Medusa generation、compare、benchmark 都会用到这个路径。

## 8. 跑 Medusa generation

### 8.1 单条 prompt

```bash
python -m pythia_medusa.generation.medusa_generate \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_small/checkpoint-epoch01-step00010 \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --greedy \
  --output outputs/medusa_generation.json
```

当前默认就是 `--verify-mode tree`，也就是：

- Medusa heads 先给出一条 candidate future-token 链
- base model 用一次 tree verification forward 来验证这条链

如果你想回退到旧的串行逐 token 验证路径，只需要显式传：

```bash
--verify-mode serial
```

### 8.2 使用内置 prompt 集

```bash
python -m pythia_medusa.generation.medusa_generate \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_small/checkpoint-epoch01-step00010 \
  --prompt-set all \
  --limit 4 \
  --max-new-tokens 32 \
  --greedy \
  --output outputs/medusa_generation_all.json
```

### 8.3 使用自定义 prompt 文件

```bash
python -m pythia_medusa.generation.medusa_generate \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_small/checkpoint-epoch01-step00010 \
  --prompt-file data/prompts.jsonl \
  --max-new-tokens 32 \
  --greedy \
  --output outputs/medusa_generation_from_file.jsonl
```

### 8.4 你会看到哪些统计

Medusa generation 输出不仅有文本，还有：

- `rounds`
- `accept_lengths`
- `average_accept_length`
- `accept_length_histogram`
- `tokens_per_sec`
- `round_traces`
- `verify_mode`

这些信息在 Phase 5 和 Phase 6 会继续被复用。

## 9. 跑对比评估

### 9.1 用配置文件

仓库里已有：

- `configs/eval/eval_lm.yaml`

运行：

```bash
python -m pythia_medusa.eval.compare_models \
  --config configs/eval/eval_lm.yaml
```

### 9.2 直接命令行

```bash
python -m pythia_medusa.eval.compare_models \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_small/checkpoint-epoch01-step00010 \
  --dataset data/valid.jsonl \
  --output-dir outputs/eval_compare \
  --prompt-set all \
  --max-new-tokens 32
```

这里如果不额外指定，`compare_models` 里的 Medusa 生成部分也默认走 `tree_verify`。

### 9.3 会输出什么

输出目录里通常有：

- `comparison_summary.json`
- `comparison_summary.csv`
- `generation_compare.jsonl`

其中会包含：

- baseline loss / perplexity
- medusa wrapper baseline-path loss / perplexity
- head-wise accuracy / loss
- baseline vs Medusa 的文本生成结果对比
- 当前使用的 `verify_mode`

## 10. 跑 benchmark

### 10.1 用配置文件

仓库里已有：

- `configs/eval/benchmark_gen.yaml`

运行：

```bash
python -m pythia_medusa.eval.benchmark_generation \
  --config configs/eval/benchmark_gen.yaml
```

### 10.2 直接命令行

```bash
python -m pythia_medusa.eval.benchmark_generation \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_small/checkpoint-epoch01-step00010 \
  --output-dir outputs/benchmark_compare \
  --prompt-set all \
  --max-new-tokens 32 \
  --repeat 5 \
  --warmup 1
```

这里默认也会走 `tree_verify`；如果你想把 benchmark 做成 “tree vs serial” 对照，只需要额外补一个：

```bash
--verify-mode serial
```

### 10.3 用 shell 脚本

```bash
bash scripts/benchmark_compare.sh
```

也可以先设环境变量再跑：

```bash
export CHECKPOINT_PATH=outputs/medusa_train_small/checkpoint-epoch01-step00010
export OUTPUT_DIR=outputs/benchmark_compare
bash scripts/benchmark_compare.sh
```

### 10.4 benchmark 输出

会生成：

- `benchmark_summary.json`
- `benchmark_summary.csv`
- `baseline_runs.jsonl`
- `medusa_runs.jsonl`

核心指标：

- `avg_latency_sec`
- `avg_tokens_per_sec`
- `avg_accept_length`
- `latency_stats`
- `throughput_stats`

## 11. 推荐实操顺序

如果你是第一次跑，最省心的顺序是：

1. 准备环境
2. 先跑 baseline generation
3. 再跑 baseline evaluation
4. 准备 `train.jsonl` 和 `valid.jsonl`
5. 开始训练 Medusa heads
6. 取训练出的 checkpoint 跑 Medusa generation
7. 跑 compare_models
8. 最后跑 benchmark

## 12. 一个最小闭环示例

### 第一步：安装环境

```bash
conda create -n pythia-medusa python=3.11 -y
conda activate pythia-medusa
pip install -r requirements.txt
```

### 第二步：准备数据

先准备数据：

```bash
python -m pythia_medusa.data.prepare_text_dataset \
  --preset wikitext-2 \
  --train-output data/train.jsonl \
  --valid-output data/valid.jsonl \
  --manifest-output data/dataset_manifest.json \
  --chunk-size 128 \
  --max-train-records 512 \
  --max-valid-records 128
```

如果你暂时不想联网，也可以自己手工写 `data/train.jsonl` / `data/valid.jsonl`，格式仍然是每行一个 `{"text": ...}`。

### 第三步：训练

```bash
python -m pythia_medusa.training.trainer \
  --model EleutherAI/pythia-70m-deduped \
  --dataset data/train.jsonl \
  --valid-dataset data/valid.jsonl \
  --output-dir outputs/medusa_train_demo \
  --seq-len 128 \
  --batch-size 2 \
  --num-epochs 1 \
  --medusa-num-heads 3 \
  --medusa-num-layers 1
```

### 第四步：生成

```bash
python -m pythia_medusa.generation.medusa_generate \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_demo/checkpoint-epoch01-step00002 \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --greedy
```

### 第五步：评估

```bash
python -m pythia_medusa.eval.compare_models \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_demo/checkpoint-epoch01-step00002 \
  --dataset data/valid.jsonl \
  --output-dir outputs/eval_demo \
  --prompt-set all \
  --max-new-tokens 16
```

### 第六步：benchmark

```bash
python -m pythia_medusa.eval.benchmark_generation \
  --model EleutherAI/pythia-70m-deduped \
  --checkpoint-path outputs/medusa_train_demo/checkpoint-epoch01-step00002 \
  --output-dir outputs/benchmark_demo \
  --prompt-set short_factual \
  --max-new-tokens 16 \
  --repeat 3 \
  --warmup 1
```

## 13. 常见问题

### 13.1 `No module named torch`

说明依赖没装好：

```bash
pip install -r requirements.txt
```

### 13.2 首次运行特别慢

大概率是在下载：

```text
EleutherAI/pythia-70m-deduped
```

首次下载慢是正常的。

### 13.3 训练能跑，但 Medusa 没明显变快

这不一定是 bug。

当前实现是 correctness-first：

- 更偏研究原型
- 更偏行为验证
- 更偏 acceptance/logging 可解释性

还没有针对 GPTNeoX 做真正深入的性能优化。

### 13.4 compare 和 benchmark 哪个先跑

建议先 `compare_models`，再 `benchmark_generation`。

因为先确认：

- loss 正常
- perplexity 正常
- head accuracy 不是异常值
- 生成结果不是空的

再去看速度，会更靠谱。

## 14. 相关文档

阶段规划文档：

- `docs/pythia_medusa_plan/phase_01_baseline.md`
- `docs/pythia_medusa_plan/phase_02_medusa_structure.md`
- `docs/pythia_medusa_plan/phase_03_medusa_generation.md`
- `docs/pythia_medusa_plan/phase_04_training.md`
- `docs/pythia_medusa_plan/phase_05_evaluation.md`
- `docs/pythia_medusa_plan/phase_06_speed_benchmark.md`

仓库总说明：

- `README.md`
