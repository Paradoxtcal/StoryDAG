"""Tests for CNGE chapter and scene segmentation."""

from storydag.cnge import Chapter, Scene, segment_into_chapters, segment_novel, segment_scenes

SAMPLE_NOVEL = """\
第一章 初入江湖

李明拔剑出鞘，寒光一闪。
"谁敢上前？"他喝道。

***

三日后，城外驿站。

王芳推门而入，神色慌张。
"出事了。"她低声说。


第二章 风雨欲来

与此同时，皇宫深处。

皇帝放下奏折，眉头紧锁。
"""


def test_segment_into_chapters_with_chinese_headers():
    chapters = segment_into_chapters(SAMPLE_NOVEL)
    assert len(chapters) == 2
    assert chapters[0][0] == "ch1"
    assert chapters[0][1] == "第一章 初入江湖"
    assert "李明拔剑出鞘" in chapters[0][2]
    assert chapters[1][0] == "ch2"
    assert chapters[1][1] == "第二章 风雨欲来"


def test_segment_into_chapters_without_headers():
    text = "没有章节标题的纯文本。\n第二段内容。"
    chapters = segment_into_chapters(text)
    assert len(chapters) == 1
    assert chapters[0] == ("ch1", "正文", text)


def test_segment_into_chapters_empty_input():
    assert segment_into_chapters("") == []
    assert segment_into_chapters("   \n  ") == []


def test_segment_scenes_by_asterisk_and_time_skip():
    chapter_text = """\
开场对白。

***

三日后，城外驿站。

第二场景内容。
"""
    scenes = segment_scenes("ch1", chapter_text)
    assert len(scenes) == 2
    assert scenes[0].scene_id == "ch1_s1"
    assert "开场对白" in scenes[0].text
    assert scenes[1].scene_id == "ch1_s2"
    assert scenes[1].text.startswith("三日后")
    assert "第二场景内容" in scenes[1].text


def test_segment_scenes_by_double_blank_line():
    chapter_text = "第一段。\n\n\n第二段。"
    scenes = segment_scenes("ch3", chapter_text)
    assert len(scenes) == 2
    assert scenes[0].text == "第一段。"
    assert scenes[1].text == "第二段。"


def test_segment_scenes_single_scene_when_no_breaks():
    chapter_text = "连续叙事，没有明显场景切换。"
    scenes = segment_scenes("ch1", chapter_text)
    assert len(scenes) == 1
    assert scenes[0].scene_id == "ch1_s1"
    assert scenes[0].text == chapter_text


def test_segment_scenes_empty_chapter_body():
    assert segment_scenes("ch1", "") == []
    assert segment_scenes("ch1", "   ") == []


def test_segment_novel_returns_chapter_objects():
    chapters = segment_novel(SAMPLE_NOVEL)
    assert len(chapters) == 2
    assert all(isinstance(ch, Chapter) for ch in chapters)
    assert chapters[0].chapter_id == "ch1"
    assert chapters[0].title == "第一章 初入江湖"
    assert len(chapters[0].scenes) >= 2
    assert all(isinstance(sc, Scene) for sc in chapters[0].scenes)
    assert chapters[0].scenes[0].scene_id == "ch1_s1"
    assert "李明拔剑出鞘" in chapters[0].scenes[0].text


def test_chapter_text_property_joins_scenes():
    chapter = Chapter(
        chapter_id="ch1",
        index=0,
        title="第一章",
        scenes=[
            Scene(scene_id="ch1_s1", index=0, text="场景一"),
            Scene(scene_id="ch1_s2", index=1, text="场景二"),
        ],
    )
    assert chapter.text == "场景一\n\n场景二"


def test_segment_novel_numeric_chapter_header():
    text = "第12章 数字标题\n\n正文内容。"
    chapters = segment_novel(text)
    assert len(chapters) == 1
    assert chapters[0].title == "第12章 数字标题"
    assert chapters[0].scenes[0].text == "正文内容。"
