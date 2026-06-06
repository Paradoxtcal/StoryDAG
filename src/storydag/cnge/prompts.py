"""Prompt templates and few-shot examples for CNGE causal triple extraction."""

from __future__ import annotations

import json
from typing import List

from storydag.llm.types import ChatMessage

SYSTEM_PROMPT = """\
你是因果叙事图抽取专家。从小说场景文本中抽取有向因果三元组。

执行步骤（chain-of-thought，先在脑中完成再输出 JSON）：
1. 识别故事实体（人物、地点、关键物件）
2. 识别叙事节点：事件(event)、意图(intention)、揭示(revelation)、情感状态(emotional_state)
3. 识别节点间的有向因果边，边类型只能是：
   - motivates：意图或目标导致行动
   - informs：信息从一方传递给另一方
   - emotionally_causes：情感事件导致Disposition变化

输出要求：
- 仅输出一个 JSON 对象，格式为 {"triples": [...]}
- 每个三元组字段：source, source_label, source_type, edge_type, target, target_label, target_type
- 节点 ID 必须使用前缀 {scene_id}_n{{序号}}，如 ch1_s2_n1
- source_type / target_type 只能是：event, intention, revelation, emotional_state
- edge_type 只能是：motivates, informs, emotionally_causes
- 不要编造文本中未出现的情节
"""

FEW_SHOT_EXAMPLES: List[tuple[str, str]] = [
    (
        'scene_id: demo_s1\n\n林远得知师妹被囚于后山。他握紧剑柄，眼神冰冷。"今夜我必救她出来。"',
        json.dumps(
            {
                "triples": [
                    {
                        "source": "demo_s1_n1",
                        "source_label": "林远得知师妹被囚于后山",
                        "source_type": "revelation",
                        "edge_type": "motivates",
                        "target": "demo_s1_n2",
                        "target_label": "林远决意今夜救人",
                        "target_type": "intention",
                    },
                    {
                        "source": "demo_s1_n2",
                        "source_label": "林远决意今夜救人",
                        "source_type": "intention",
                        "edge_type": "emotionally_causes",
                        "target": "demo_s1_n3",
                        "target_label": "林远眼神冰冷、情绪紧绷",
                        "target_type": "emotional_state",
                    },
                ]
            },
            ensure_ascii=False,
        ),
    ),
    (
        'scene_id: demo_s2\n\n掌柜压低声音："令牌在城西当铺。" 阿青一愣，随即点头。',
        json.dumps(
            {
                "triples": [
                    {
                        "source": "demo_s2_n1",
                        "source_label": "掌柜告知令牌在城西当铺",
                        "source_type": "event",
                        "edge_type": "informs",
                        "target": "demo_s2_n2",
                        "target_label": "阿青获知令牌下落",
                        "target_type": "revelation",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ),
    (
        "scene_id: demo_s3\n\n父亲病逝的消息传来，沈砚跪倒在地，久久不语。三日后，他撕毁婚书。",
        json.dumps(
            {
                "triples": [
                    {
                        "source": "demo_s3_n1",
                        "source_label": "沈砚得知父亲病逝",
                        "source_type": "revelation",
                        "edge_type": "emotionally_causes",
                        "target": "demo_s3_n2",
                        "target_label": "沈砚悲痛失语",
                        "target_type": "emotional_state",
                    },
                    {
                        "source": "demo_s3_n2",
                        "source_label": "沈砚悲痛失语",
                        "source_type": "emotional_state",
                        "edge_type": "motivates",
                        "target": "demo_s3_n3",
                        "target_label": "沈砚撕毁婚书",
                        "target_type": "event",
                    },
                ]
            },
            ensure_ascii=False,
        ),
    ),
]


def build_extraction_messages(scene_id: str, scene_text: str) -> List[ChatMessage]:
    """Build system + few-shot + user messages for one scene chunk."""
    system = SYSTEM_PROMPT.replace("{scene_id}", scene_id)
    messages: List[ChatMessage] = [ChatMessage(role="system", content=system)]

    for user_example, assistant_example in FEW_SHOT_EXAMPLES:
        messages.append(ChatMessage(role="user", content=user_example))
        messages.append(ChatMessage(role="assistant", content=assistant_example))

    user_content = f"scene_id: {scene_id}\n\n{scene_text.strip()}"
    messages.append(ChatMessage(role="user", content=user_content))
    return messages
