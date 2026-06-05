# StoryDAG

基于**硬因果约束**与**因果历史条件化角色模块**的长篇小说自动改编剧本系统。

## 问题

直接将长篇小说（≥3 章）翻译为结构化剧本时，现有管线系统性地出现两类致命缺陷：

1. **角色崩塌**：剧本中人物的行为、台词与动机，与小说中已建立的人格弧线和情感状态不可逆地矛盾。
2. **因果断裂**：场景顺序丢失因果链，关键揭示出现在前提信息建立之前。

StoryDAG 将改编重构为**因果连续性保持的弹性图重写问题**：先抽取可审计的因果 DAG，再在硬拓扑序约束下优化场景排序，并以因果历史子图约束角色生成。

## 架构

```
小说文本
   │
   ▼
┌─────────────────────────────────────┐
│  CNGE  因果叙事图抽取               │
│  事件/意图/揭示节点 + 类型化边       │
└─────────────────────────────────────┘
   │ causal_graph.json
   ▼
┌─────────────────────────────────────┐
│  CausOpt  硬约束场景排序 (MCTS-HP)  │
│  保证 ∀(A→B): scene(A) < scene(B)   │
└─────────────────────────────────────┘
   │ scene_sequence
   ▼
┌─────────────────────────────────────┐
│  CGCA  因果门控角色生成             │
│  logit 修正 + 硬黑名单              │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  Serialization  可验证 YAML 输出  │
│  satisfied_edges / causal_backlink  │
│  → CCR (Causal Closure Ratio)       │
└─────────────────────────────────────┘
```

## 模块

| 模块 | 路径 | 职责 |
|------|------|------|
| CNGE | `src/storydag/cnge/` | LLM 抽取因果三元组 → 共指消解 → DAG 构建 |
| CausOpt | `src/storydag/causopt/` | MCTS 硬剪枝场景排序，优化三幕结构与节奏 |
| CGCA | `src/storydag/cgca/` | 因果历史子图约束的角色台词/动作生成 |
| Serialization | `src/storydag/serialization/` | YAML 序列化 + CCR 自动度量 |

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # 填入 API Key
```

## 快速验证

```bash
pytest tests/test_config.py -v
```

## 数据格式

- 因果图 schema：`schemas/causal_graph.json`
- 剧本输出 schema：`schemas/script_output.yaml`

## 参考文档

改编方法论见 `docs/references/`（从上游 novel-to-script-team 精选保留）。

## 许可证

MIT — 衍生自 [novel-to-script-team](https://github.com/novel-to-script-team/novel-to-script-team)。
