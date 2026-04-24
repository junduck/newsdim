# newsdim

基于 BGE Embedding + 线性头的轻量级新闻维度标注器。对中文财经新闻在 8 个投资行为维度上打分。

## 快速开始

**Python 库**

```bash
git clone https://github.com/junduck/newsdim
cd newsdim
uv sync
```

```python
from newsdim import Tagger

tagger = Tagger()  # 首次运行自动下载 BGE 模型 (~400MB)
scores = tagger.score("煤炭板块盘初走强，大有能源涨停")
print(scores.to_dict())
# {'mom': 2, 'stab': 0, 'horz': -1, 'eng': 1, 'hype': 1, 'sent': 1, 'sec': 0, 'pol': 0}
```

**HTTP 服务**

```bash
# 本地
uv run python -m newsdim.server

# 或 Docker
docker compose up -d
```

访问 `http://localhost:8427/docs` 查看交互式 API 文档。

```bash
curl -X POST http://localhost:8427/score \
  -H "Content-Type: application/json" \
  -d '{"text": "央行宣布降准50个基点"}'
# {"scores": {"mom": 0, "stab": 0, "horz": 1, "eng": 2, "hype": 1, "sent": 0, "sec": 3, "pol": 2}}
```

---

## 维度定义

| 代码 | 名称 | 说明 |
|------|------|------|
| `mom` | 动势 | 价格趋势方向：+3 强势突破 → -3 暴跌破位 |
| `stab` | 稳定性 | 经营可预测性：+3 现金牛 → -3 暴雷退市 |
| `horz` | 时间尺度 | 投资逻辑周期：+3 长期结构性变革 → -3 纯短线事件 |
| `eng` | 事件活跃度 | 事件本身的重要程度：+3 重大重组 → 0 无事件（≥0） |
| `hype` | 关注度 | 市场讨论热度：+3 全民热议 → 0 无关注度信号（≥0） |
| `sent` | 情绪 | 正负情绪：+2 乐观 → -2 悲观 |
| `sec` | 范围 | 影响范围：+3 宏观/全市场 → -3 单一公司 |
| `pol` | 政策 | 政策关联度：+3 重大政策利好 → -3 政策利空 |

分数范围 -3 ~ +3 整数，大部分维度默认为 0。

---

## 实现细节

### 架构

```
文本 → BGE-base-zh-v1.5 (768维, 冻结) → 线性头 (768→8) → 8维整数分数
```

- 嵌入模型：`BAAI/bge-base-zh-v1.5`，L2 归一化，冻结不训练
- 线性头：岭回归最小二乘法训练（ridge=1.0）
- 推理：单次矩阵乘法，无 GPU 亦可

### 性能

基于 25,000 篇新闻训练，10% 验证集评估：

| 维度 | 符号一致率 | MAE | Kendall's τ |
|------|-----------|-----|-------------|
| mom  | 91.9% | 0.455 | 0.590 |
| stab | 90.6% | 0.454 | 0.482 |
| horz | 94.7% | 0.434 | 0.678 |
| eng  | 99.9% | 0.398 | 0.557 |
| hype | 99.4% | 0.347 | 0.551 |
| sent | 93.1% | 0.500 | 0.669 |
| sec  | 93.6% | 0.767 | 0.726 |
| pol  | 90.4% | 0.428 | 0.479 |
| **整体** | **94.2%** | **0.473** | — |

### HTTP 服务

自定义端口：

```bash
uv run uvicorn newsdim.server:app --host 0.0.0.0 --port 9000
```

接口：

- `POST /score` — `{"text": "..."}` → `{"scores": {...}}`
- `POST /score/batch` — `{"texts": ["...", "..."]}` → `{"results": [{...}, ...]}`

### API 参考

```python
from newsdim import Tagger, DIMS, DimScores, DIM_LABELS

tagger = Tagger(weights_path=None, device=None)
tagger.score(text: str) -> DimScores
tagger.score_batch(texts: list[str], batch_size: int = 64) -> list[DimScores]
tagger.score_raw(text: str) -> dict[str, float]
tagger.score_batch_raw(texts: list[str]) -> list[dict[str, float]]

s = DimScores(mom=1, stab=-2)
s.to_dict()    # {'mom': 1, 'stab': -2, 'horz': 0, ...}
s.to_array()   # [1, -2, 0, ...]
DimScores.from_dict(d)
DimScores.from_array(arr)
```

### 代码结构

```
src/newsdim/
├── __init__.py          # 公开 API：Tagger, DIMS, DimScores
├── dims.py              # 维度定义与 DimScores 数据类
├── tagger.py            # Tagger 推理类
├── server.py            # FastAPI 服务
├── assets/              # 训练好的权重文件
├── embed/               # BGE 编码器封装
├── train/               # 训练逻辑与评估指标
├── news_scorer/         # LLM 标注工具（用于生成训练数据）
└── ann_scorer/          # 公告规则打分（内部模块）

scripts/                 # 数据处理与训练流水线脚本
tests/                   # 测试
```

### 训练流水线

```bash
uv run python scripts/ingest.py                        # 数据入库
uv run python scripts/score_news.py --limit 1000       # LLM 打分
uv run python scripts/precompute_embeddings.py         # 预计算嵌入
uv run python scripts/train_head.py --ridge 1.0        # 训练
uv run python scripts/evaluate_head.py                 # 评估
uv run python scripts/data_quality.py                  # 数据质量报告
```
