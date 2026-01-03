from __future__ import annotations

from typing import List, Optional


# === 默认提示词常量（集中管理，可供用户修改） ===

DEFAULT_SYSTEM_PROMPT = (
    "你是一个中文助理，擅长从被引用的消息中提炼含义、意图和注意事项。"
    "请严格使用如下 Markdown 结构输出回答：\n"
    '1. 第一行以"关键词："开头，列出 3~8 个核心关键词，使用" | "分隔，例如：\n'
    "   关键词：chcp 936 | 代码页 | GBK | 批处理 | 乱码\n"
    "2. 接着给出 1~2 句总括性说明，可以单独成段。\n"
    '3. 然后输出一行"**详细阐述：**"，在其后用若干段落进行详细解释。\n'
    "**内容安全审查**：涉及暴力、争议、不雅内容，必须明确且礼貌地拒绝。\n"
    "**绝对红线**：涉及政治、历史人物评价、R18、人身伤害内容，立即停止思考并输出拒绝话术。\n"
    "禁止输出思考过程或中间推理，只保留对用户有用的结论性内容。"
)

DEFAULT_TEXT_USER_PROMPT = (
    "请解释这条被回复的消息的含义，输出简洁不超过100字。禁止输出政治有关内容。\n"
    "原始文本：\n{text}"
)

DEFAULT_IMAGE_USER_PROMPT = (
    "请解释这条被回复的消息/图片的含义，输出简洁不超过100字。禁止输出政治有关内容。\n"
    "{text_block}\n包含图片：若无法直接读取图片，请结合上下文或文件名描述。"
)


def build_user_prompt(text: Optional[str], images: List[str]) -> str:
    """根据是否包含图片选择文本/图文提示词模板。"""
    text_block = ("原始文本:\n" + text) if text else ""
    tmpl = DEFAULT_IMAGE_USER_PROMPT if images else DEFAULT_TEXT_USER_PROMPT
    return tmpl.format(text=text or "", text_block=text_block)


def build_system_prompt() -> str:
    """返回系统提示词（供 LLM 调用使用）。"""
    return DEFAULT_SYSTEM_PROMPT


def build_system_prompt_for_event(
    custom_persona_setting: str = "",
    override_enabled: bool = False,
) -> str:
    """根据配置构造系统提示词。
    
    Args:
        custom_persona_setting: 用户自定义的人格设定文本。
        override_enabled: 是否启用人格覆写开关。
        
    Returns:
        如果 override_enabled 为 True 且 custom_persona_setting 非空，则返回自定义人格；
        否则返回默认系统提示词。
    """
    if override_enabled and custom_persona_setting.strip():
        return custom_persona_setting.strip()
    return DEFAULT_SYSTEM_PROMPT
