# StoryDAG

**Causal-continuity-preserving novel-to-script adaptation via hard DAG constraints.**

将长篇小说自动转化为结构化的、因果可审计的多幕剧本。管线由四个算法模块构成，每个模块产出可独立验证的中间工件。

---

## Architecture

```
novel.txt
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  CNGE · Causal Narrative Graph Extraction                    │
│  小说分段 → LLM 因果三元组抽取 → 共指消解 → DAG 构建     │
│  输出: causal_graph.json                                     │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  CausOpt · Hard-Constrained Scene Ordering                   │
│  MCTS-HP 硬剪枝搜索 → 三幕场景排序 → 边满足度矩阵         │
│  输出: SceneSequence (场景列表 + satisfied_edges)            │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  CGCA · Causal-Gated Character Adapter                       │
│  逐场景角色台词生成 · 因果历史子图提取 · 软门控 + 硬黑名单 │
│  输出: 每场景 ScriptBeat[]                                   │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Serialization · Verifiable Script Artifacts                  │
│  YAML 剧本序列化 · CCR 自动度量 · Plot hole 检测            │
│  输出: script.yaml + metrics.json                             │
└──────────────────────────────────────────────────────────────┘
```

输出目录结构：

```
outputs/{title}/
├── causal_graph.json    # CNGE 因果 DAG
├── script.yaml          # 可验证的多幕剧本
└── metrics.json         # CCR 与因果一致性报告
```

---

## Schema Design Rationale

### Why two schemas? — `causal_graph.json` + `script_output.yaml`

因果 DAG 和剧本 YAML 分属两个不同的抽象层级，用一个 schema 承载会导致语义耦合和版本僵化。

**[`schemas/causal_graph.json`](schemas/causal_graph.json)** 是 **分析层** 的产物，描述"故事世界中的因果事实"——节点是叙事事件/意图/揭示，边是它们之间的因果关系。此 schema 稳定：不论剧本最终如何排列场景、分配台词，因果事实不变。节点类型（`revelation` / `intention` / `event` / `emotional_state`）和边类型（`motivates` / `informs` / `emotionally_causes`）直接对应算法论文的定义，确保从 CNGE 抽取到 CausOpt 优化的无损传递。

**[`schemas/script_output.yaml`](schemas/script_output.yaml)** 是 **创作层** 的产物，描述"剧本中说了什么、谁说的、基于什么因果授权"。每个字段的存在理由如下：

| 字段 | 存在原因 |
|------|----------|
| `scene.satisfied_edges` | **CCR 可计算性** — 无需语义理解，仅凭边 ID 列表即可计算 Causal Closure Ratio：满足边数 / 总边数。这使得因果完整性成为可自动度量的工程指标，而非文学评论。 |
| `scene.assigned_node_ids` | **可逆性** — 建立剧本场景 ↔ 因果图节点的双向链接。当因果图被人工修改后，可据此定位受影响的场景进行局部重生成，无需重跑全管线。 |
| `narrative_time_clue` | **叙事时序审计** — 保留节点在 DAG 拓扑序中的位置范围。剧本的呈现顺序（scene index）可与叙事时间（narrative time）不一致（如倒叙），此字段为编辑提供原始因果时序证据。 |
| `line.causal_backlink` | **角色一致性审计** — 每条台词/动作声称的因果授权边。PR #14 据此逐条验证：角色所知道/所动机的因果边是否在拓扑序上不晚于当前场景。防止"角色提前知道不该知道的事"。 |
| `beats[]` | **编辑粒度** — 比 scene 更细的节拍单元，支持同一场景内的多轮对话独立审核与替换。 |
| `metadata.causopt_score` | **可复现性** — MCTS 最优分数与来源图路径，使剧本版本可追溯。 |

核心设计原则：**每一个 schema 字段都应该对应一种无需 LLM 的自动验证**。能通过 ID 列表而非自然语言表达的约束，就不应该依赖语义理解。

---

## Module Documentation

### CNGE — Causal Narrative Graph Extraction

| 步骤 | 实现 | 说明 |
|------|------|------|
| Chapter Segmentation | `segmentation.segment_novel()` | 正则匹配 `第X章` 标题，若无则整篇作为一章 |
| Scene Segmentation | `segmentation.segment_scenes()` | 空行、分隔线、时间跳跃标记（`第二天`、`数年后` 等）作为场景边界 |
| Triple Extraction | `extractor.extract_triples()` | 每场景送 LLM，`json_mode=True` 获取结构化三元组，枚举类型校验 |
| Coreference Resolution | `coref.resolve_coreferences()` | sentence-transformers 编码节点 label，余弦相似度 >0.85 union-find 聚类合并 |
| DAG Construction | `graph.build_dag()` | networkx 构建有向图，迭代移除最弱边直到无环 + topological_sort |

### CausOpt — Hard-Constrained Scene Ordering

| 组件 | 实现 | 说明 |
|------|------|------|
| Hard Constraints | `models.is_valid_assignment()` | 边 `u→v` 要求 `scene_index(u) < scene_index(v)`，MCTS 层合法性剪枝 |
| MCTS Search | `mcts.search()` | UCT 树搜索，`cluster_ready_nodes()` 拓扑邻居聚类控制分支因子 |
| Scoring | `scoring.evaluate_assignment()` | 戏剧曲线 + 幕比例 + 角色均匀性 + 叙事节奏加权的多目标函数 |
| Rollout | `rollout.greedy_rollout()` | 拓扑序贪心补全未分配节点 |

### CGCA — Causal-Gated Character Adapter

| 组件 | 实现 | 说明 |
|------|------|------|
| History Extraction | `history.extract_history()` | H(c,S) 子图：history nodes = ancestor(scene_nodes) + scene_nodes |
| Soft Gating | `gating.apply_gate_to_logits()` | token 级因果门控：未来因果节点不应出现在当前台词中 |
| Hard Blacklist | `blacklist.find_unknown_secrets()` | 角色未知的因果节点作为黑名单，token 级硬屏蔽 |
| Generation | `generator.generate_character_line()` | LLM 调用 + 后检查硬黑名单，最多 retry 3 次 |

### Serialization — Verifiable Script Artifacts

| 组件 | 实现 | 说明 |
|------|------|------|
| YAML Writer | `yaml_writer.build_script()` | SceneSequence + ScriptBeat → ScriptYAML → yaml.safe_dump |
| YAML Reader | `yaml_reader.read_script()` | 逆转 + validate：字段完整性、`satisfied_edges` ID 合法性、边拓扑序 |
| Metrics | `metrics.compute_metrics()` | CCR、Causal Density、Character Consistency、Plot Hole、Backlink 五项审计 |
| CLI | `python -m storydag run --novel path.txt --title "剧本名"` | 完整管线 |

---

## CCR & Metrics

**Causal Closure Rate (CCR)** = `satisfied_edge_count / total_edges`

| 指标 | 含义 |
|------|------|
| `ccr` | 剧本满足的因果边比例。1.0 = 所有因果承诺均已兑现 |
| `causal_density` | 每场景满足的边数。过高意味场景承载过多因果节点，可能节奏失衡 |
| `character_consistency` | 每角色台词 `causal_backlink` 是否合乎 DAG 拓扑序 |
| `plot_holes` | 因果图中有边但剧本从未满足的边 ID 列表 |
| `out_of_order_edges` | 边的 target 节点出现在 source 节点之前的场景 |
| `backlink_violations` | 台词引用了一条当前场景尚未满足的因果边（角色"提前知道"） |

---

## Installation

```bash
git clone https://github.com/Paradoxtcal/StoryDAG.git
cd StoryDAG

python -m venv .venv
source .venv/bin/activate

pip install -e .
# 可选：本地 LLM 支持
pip install -e ".[local-llm]"
```

配置 `.env`（参照 [`.env.example`](.env.example)）：

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4-turbo
EMBEDDING_MODEL=all-MiniLM-L6-v2
MCTS_ITERATIONS=10000
MCTS_TIME_BUDGET_SEC=300
```

---

## Usage

### CLI — Full Pipeline

```bash
python -m storydag run --novel novel.txt --title "我的剧本"
# 产出 outputs/我的剧本/{causal_graph.json, script.yaml, metrics.json}
```

### Python API

```python
from storydag.pipeline import run_pipeline

result = run_pipeline("novel.txt", "我的剧本")
print(result.metrics.ccr)               # 0.92
print(result.sequence.scenes)           # [SceneRecord, ...]
print(result.script.acts[0].scenes[0])  # ScriptScene
```

### Flask Audit Interface

```bash
# 加载 outputs/ 下的因果图
python -m storydag.cnge.auditor.app

# 或直接指定图文件
python -m storydag.cnge.auditor.app outputs/我的剧本/causal_graph.json
```

在 `http://127.0.0.1:5000` 打开可视化面板，支持：

- **图谱拖拽**：vis.js 力导向布局，逐节点拖放重排
- **节点编辑**：点击任意节点编辑 label / type，即时更新画布
- **边编辑**：点击任意边修改 type / strength
- **增删操作**：工具栏"添加节点""添加边"模式，点击画布空白/节点即可创建
- **保存回写**：编辑后"保存"按钮写回 `causal_graph.json`，拓扑序自动重建

---

## Tests

```bash
pytest tests/ -q
# 92 tests passed (all 4 modules + pipeline + metrics)
```

---

## License

MIT
