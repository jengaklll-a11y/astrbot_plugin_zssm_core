from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Set, Union
import os
import asyncio
import re
import shutil
import time

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import astrbot.api.message_components as Comp
from astrbot.core.star.star_handler import EventType
from astrbot.core.pipeline.context_utils import call_event_hook

from .message_utils import (
    extract_quoted_payload,
    extract_text_and_images_from_chain,
    call_get_msg,
    ob_data,
    napcat_resolve_file_url,
    extract_from_onebot_message_payload,
)
from .prompt_utils import (
    build_user_prompt,
    build_system_prompt_for_event,
)
from .llm_client import LLMClient
from .file_preview_utils import (
    build_text_exts_from_config,
    extract_file_preview_from_reply,
)

# 配置 Key
KEYWORD_ZSSM_ENABLE_KEY = "enable_keyword_zssm"
FILE_PREVIEW_EXTS_KEY = "file_preview_exts"
FILE_PREVIEW_MAX_SIZE_KB_KEY = "file_preview_max_size_kb"
SEARCH_PROVIDER_ID_KEY = "search_provider_id"
PERSONA_SETTING_KEY = "persona_setting"
PERSONA_OVERRIDE_ENABLED_KEY = "persona_override_enabled"

# 默认值
DEFAULT_FILE_PREVIEW_EXTS = "txt,md,log,json,csv,ini,cfg,yml,yaml,py"
DEFAULT_FILE_PREVIEW_MAX_SIZE_KB = 100


@register(
    "astrbot_plugin_zssm_core",
    "jengaklll-a11y",
    '可直接zssm解释引用内容，或使用 "zssm + 内容" 进行联网搜索',
    "1.1.0",
    "https://github.com/jengaklll-a11y/astrbot_plugin_zssm_core",
)
class ZssmExplain(Star):
    def __init__(self, context: Context, config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config: Dict[str, Any] = config or {}
        self._llm = LLMClient(
            context=self.context,
            get_conf_int=self._get_conf_int,
            get_config_provider=self._get_config_provider,
            logger=logger,
        )

    async def initialize(self):
        """可选：插件初始化。"""
        pass

    def _reply_text_result(self, event: AstrMessageEvent, text: str):
        """构造纯文本消息结果。"""
        safe_text = str(text).strip() if text is not None else ""
        return event.plain_result(safe_text)

    def _get_conf_str(self, key: str, default: str) -> str:
        try:
            v = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(v, str):
                return v.strip()
        except Exception:
            pass
        return default

    def _get_conf_bool(self, key: str, default: bool) -> bool:
        try:
            v = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                lv = v.strip().lower()
                if lv in ("1", "true", "yes", "on"):
                    return True
                if lv in ("0", "false", "no", "off"):
                    return False
        except Exception:
            pass
        return default

    def _get_conf_int(
        self, key: str, default: int, min_v: int = 1, max_v: int = 120000
    ) -> int:
        try:
            v = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(v, int):
                return max(min(v, max_v), min_v)
            if isinstance(v, str) and v.strip().isdigit():
                return max(min(int(v.strip()), max_v), min_v)
        except Exception:
            pass
        return default

    @staticmethod
    def _is_zssm_trigger(text: str) -> bool:
        if not isinstance(text, str):
            return False
        t = text.strip()
        if re.match(r"^[\s/!！。\.、，\-]*zssm(\s|$)", t, re.I):
            return True
        return False

    @staticmethod
    def _first_plain_head_text(chain: List[object]) -> str:
        if not isinstance(chain, list):
            return ""
        for seg in chain:
            try:
                if isinstance(seg, Comp.Plain):
                    txt = getattr(seg, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        return txt
            except (AttributeError, TypeError):
                continue
        return ""

    @staticmethod
    def _chain_has_at_me(chain: List[object], self_id: str) -> bool:
        if not isinstance(chain, list):
            return False
        for seg in chain:
            try:
                if isinstance(seg, Comp.At):
                    qq = getattr(seg, "qq", None)
                    if qq is not None and str(qq) == str(self_id):
                        return True
            except (AttributeError, TypeError):
                continue
        return False

    def _already_handled(self, event: AstrMessageEvent) -> bool:
        try:
            extras = event.get_extra()
            if isinstance(extras, dict) and extras.get("zssm_handled"):
                return True
        except Exception:
            pass
        try:
            event.set_extra("zssm_handled", True)
        except Exception:
            pass
        return False

    @staticmethod
    def _strip_trigger_and_get_content(text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.strip()
        m = re.match(r"^[\s/!！。\.、，\-]*zssm(?:\s+(.+))?$", t, re.I)
        if not m:
            return ""
        content = (m.group(1) or "").strip()
        try:
            content = re.sub(
                r"[\[【](图片|image|img|文件|file)[\]】]",
                " ",
                content,
                flags=re.I,
            )
        except Exception:
            pass
        try:
            content = re.sub(r"\s{2,}", " ", content).strip()
        except Exception:
            content = content.strip()
        return content

    def _get_inline_content(self, event: AstrMessageEvent) -> str:
        try:
            chain = event.get_messages()
        except Exception:
            chain = (
                getattr(event.message_obj, "message", [])
                if hasattr(event, "message_obj")
                else []
            )
        head = self._first_plain_head_text(chain)
        if head:
            c = self._strip_trigger_and_get_content(head)
            if c:
                return c
        try:
            s = event.get_message_str()
        except Exception:
            s = getattr(event, "message_str", "") or ""
        return self._strip_trigger_and_get_content(s)

    @staticmethod
    def _safe_get_chain(event: AstrMessageEvent) -> List[object]:
        try:
            return event.get_messages()
        except Exception:
            return (
                getattr(event.message_obj, "message", [])
                if hasattr(event, "message_obj")
                else []
            )

    def _extract_images_from_event(self, event: AstrMessageEvent) -> List[str]:
        chain = self._safe_get_chain(event)
        try:
            _t, images = extract_text_and_images_from_chain(chain)
        except Exception:
            images = []
        return [x for x in images if isinstance(x, str) and x]

    async def _resolve_images_for_llm(
        self, event: AstrMessageEvent, images: List[str]
    ) -> List[str]:
        def _norm(x: object) -> Optional[str]:
            if not isinstance(x, str) or not x:
                return None
            s = x.strip()
            if not s:
                return None
            ls = s.lower()
            if ls.startswith(("http://", "https://")):
                return s
            if ls.startswith("base64://") or ls.startswith("data:image/"):
                return s
            if ls.startswith("file://"):
                try:
                    fp = s[7:]
                    if fp.startswith("/") and len(fp) > 3 and fp[2] == ":":
                        fp = fp[1:]
                    if fp and os.path.exists(fp):
                        return os.path.abspath(fp)
                except Exception:
                    return None
                return None
            try:
                if os.path.exists(s):
                    return os.path.abspath(s)
            except Exception:
                return None
            return None

        resolved: List[str] = []
        seen = set()

        def _add(cand: str) -> None:
            if cand and cand not in seen:
                seen.add(cand)
                resolved.append(cand)

        resolve_candidates: List[str] = []
        for img in images:
            if not isinstance(img, str) or not img:
                continue
            direct = _norm(img)
            if direct:
                _add(direct)
            else:
                resolve_candidates.append(img)

        unresolved: List[str] = []
        if resolve_candidates:
            sem = asyncio.Semaphore(6)

            async def _resolve_one(fid: str) -> Optional[str]:
                async with sem:
                    try:
                        return await napcat_resolve_file_url(event, fid)
                    except Exception:
                        return None

            tasks = [_resolve_one(fid) for fid in resolve_candidates]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for fid, res in zip(resolve_candidates, results):
                if isinstance(res, Exception) or not isinstance(res, str) or not res:
                    unresolved.append(fid)
                    continue
                rr = _norm(res)
                if rr:
                    _add(rr)
                else:
                    unresolved.append(fid)

        if unresolved and hasattr(event, "message_obj"):
            try:
                mid = getattr(event.message_obj, "message_id", None)
                mid = str(mid) if mid is not None else ""
            except Exception:
                mid = ""
            if mid:
                try:
                    ret = await call_get_msg(event, mid)
                    data = ob_data(ret or {})
                    _t, imgs2 = extract_from_onebot_message_payload(data)
                    for x in imgs2:
                        nx = _norm(x)
                        if nx:
                            _add(nx)
                except Exception as e:
                    logger.debug(
                        "zssm_explain: get_msg fallback for current images failed: %s",
                        e,
                    )

        return resolved

    def _get_file_preview_exts(self) -> Set[str]:
        raw = self._get_conf_str(FILE_PREVIEW_EXTS_KEY, DEFAULT_FILE_PREVIEW_EXTS)
        base_default = [
            ext.strip() for ext in DEFAULT_FILE_PREVIEW_EXTS.split(",") if ext.strip()
        ]
        return build_text_exts_from_config(raw, base_default)

    def _get_file_preview_max_bytes(self) -> Optional[int]:
        try:
            kb = self._get_conf_int(
                FILE_PREVIEW_MAX_SIZE_KB_KEY,
                DEFAULT_FILE_PREVIEW_MAX_SIZE_KB,
                1,
                1024 * 1024,
            )
        except Exception:
            kb = DEFAULT_FILE_PREVIEW_MAX_SIZE_KB
        try:
            return int(kb) * 1024
        except Exception:
            return None

    def _build_system_prompt(self, event: AstrMessageEvent) -> str:
        """构建系统提示词，根据开关决定是否使用自定义人格。"""
        override_enabled = self._get_conf_bool(PERSONA_OVERRIDE_ENABLED_KEY, False)
        custom_persona = self._get_conf_str(PERSONA_SETTING_KEY, "")
        return build_system_prompt_for_event(
            custom_persona_setting=custom_persona,
            override_enabled=override_enabled,
        )

    def _format_explain_output(
        self,
        content: str,
        elapsed_sec: Optional[float] = None,
    ) -> str:
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        
        body = content.strip()
        
        if not body:
            return ""

        parts: List[str] = [body]
        if isinstance(elapsed_sec, (int, float)) and elapsed_sec > 0:
            parts.append("")
            parts.append(f"cost: {elapsed_sec:.3f}s")

        return "\n".join(parts)

    def _get_config_provider(self, key: str) -> Optional[Any]:
        try:
            pid = self.config.get(key) if isinstance(self.config, dict) else None
            if isinstance(pid, str):
                pid = pid.strip()
            if pid:
                try:
                    return self.context.get_provider_by_id(provider_id=pid)
                except Exception as e:
                    logger.warning(
                        f"zssm_explain: provider id not found for {key}={pid}: {e}"
                    )
        except Exception:
            pass
        return None

    @dataclass
    class _LLMPlan:
        user_prompt: str
        images: List[str] = field(default_factory=list)
        cleanup_paths: List[str] = field(default_factory=list)
        is_search: bool = False

    @dataclass
    class _ReplyPlan:
        message: str
        stop_event: bool = True
        cleanup_paths: List[str] = field(default_factory=list)

    _ExplainPlan = Union[_LLMPlan, _ReplyPlan]

    async def _build_explain_plan(
        self,
        event: AstrMessageEvent,
        *,
        inline: str,
    ) -> _ExplainPlan:
        cleanup_paths: List[str] = []

        q_text, q_images, from_forward = await extract_quoted_payload(event)
        
        current_images_raw = self._extract_images_from_event(event)
        try:
            current_images = await self._resolve_images_for_llm(event, current_images_raw)
        except Exception:
            current_images = []
        
        all_images = (q_images or []) + current_images
        all_images = list(dict.fromkeys(all_images))

        try:
            file_preview = await extract_file_preview_from_reply(
                event,
                text_exts=self._get_file_preview_exts(),
                max_size_bytes=self._get_file_preview_max_bytes(),
            )
            if file_preview:
                q_text = f"{file_preview}\n\n{q_text}" if q_text else file_preview
        except Exception:
            pass

        if inline:
            prompt = inline
            context_str = ""
            if q_text:
                context_str += f"\n引用文本：\n{q_text}"
            
            if context_str:
                prompt += f"\n\n【上下文信息】{context_str}"

            return self._LLMPlan(
                user_prompt=prompt,
                images=all_images,
                cleanup_paths=cleanup_paths,
                is_search=True
            )

        if q_text or all_images:
            user_prompt = build_user_prompt(q_text, all_images)
            return self._LLMPlan(
                user_prompt=user_prompt,
                images=all_images,
                cleanup_paths=cleanup_paths,
                is_search=False
            )

        return self._ReplyPlan(
            message="请输入要解释的内容，或回复一条消息/图片/文件进行解释。\n使用 'zssm + 内容' 可进行联网搜索。",
            stop_event=True,
            cleanup_paths=cleanup_paths,
        )

    async def _execute_explain_plan(self, event: AstrMessageEvent, plan: _ExplainPlan):
        """执行解释计划。"""
        if isinstance(plan, self._ReplyPlan):
            yield self._reply_text_result(event, plan.message)
            if plan.stop_event:
                try:
                    event.stop_event()
                except Exception:
                    pass
            return

        user_prompt = plan.user_prompt
        images = plan.images
        is_search = plan.is_search

        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception as e:
            logger.error(f"zssm_explain: get provider failed: {e}")
            provider = None

        if not provider:
            yield self._reply_text_result(
                event, "未检测到可用的大语言模型提供商，请先在 AstrBot 配置中启用。"
            )
            return

        system_prompt = self._build_system_prompt(event)
        
        image_urls = self._llm.filter_supported_images(images)

        try:
            start_ts = time.perf_counter()
            
            if is_search:
                call_provider = self._llm.select_search_provider(
                    session_provider=provider,
                    search_provider_key=SEARCH_PROVIDER_ID_KEY
                )
            else:
                call_provider = self._llm.select_primary_provider(
                    session_provider=provider, image_urls=image_urls
                )

            llm_resp = await self._llm.call_with_fallback(
                primary=call_provider,
                session_provider=provider,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                image_urls=image_urls,
            )

            try:
                await call_event_hook(event, EventType.OnLLMResponseEvent, llm_resp)
            except Exception:
                pass

            reply_text = None
            try:
                ct = getattr(llm_resp, "completion_text", None)
                if isinstance(ct, str) and ct.strip():
                    reply_text = ct.strip()
            except Exception:
                reply_text = None
            if not reply_text:
                reply_text = self._llm.pick_llm_text(llm_resp)

            elapsed = None
            try:
                elapsed = time.perf_counter() - start_ts
            except Exception:
                elapsed = None
            reply_text = self._format_explain_output(reply_text, elapsed_sec=elapsed)
            yield self._reply_text_result(event, reply_text)
            try:
                event.stop_event()
            except Exception:
                pass
        except asyncio.TimeoutError:
            yield self._reply_text_result(
                event, "请求超时，请稍后重试或换一个模型提供商。"
            )
            try:
                event.stop_event()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"zssm_explain: LLM 调用失败: {e}")
            yield self._reply_text_result(
                event, "处理失败：模型调用异常，请稍后再试或联系管理员。"
            )
            try:
                event.stop_event()
            except Exception:
                pass

    @filter.command("zssm", alias={"知识说明", "解释"})
    async def zssm(self, event: AstrMessageEvent):
        """解释被回复消息或进行搜索。"""
        cleanup_paths: List[str] = []
        try:
            if self._already_handled(event):
                return

            inline = self._get_inline_content(event)

            plan = await self._build_explain_plan(
                event, inline=inline
            )
            try:
                cleanup_paths = list(getattr(plan, "cleanup_paths", []) or [])
            except Exception:
                cleanup_paths = []

            async for r in self._execute_explain_plan(event, plan):
                yield r
        except Exception as e:
            logger.error("zssm_explain: handler crashed: %s", e)
            yield self._reply_text_result(
                event, "解释失败：插件内部异常，请稍后再试或联系管理员。"
            )
            try:
                event.stop_event()
            except Exception:
                pass
        finally:
            try:
                for p in cleanup_paths:
                    try:
                        if isinstance(p, str) and p:
                            if os.path.isdir(p):
                                shutil.rmtree(p, ignore_errors=True)
                            elif os.path.isfile(p):
                                os.remove(p)
                    except Exception:
                        continue
            except Exception:
                pass

    async def terminate(self):
        return

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def keyword_zssm(self, event: AstrMessageEvent):
        """关键词触发。"""
        if not self._get_conf_bool(KEYWORD_ZSSM_ENABLE_KEY, True):
            return

        try:
            chain = event.get_messages()
        except Exception:
            chain = (
                getattr(event.message_obj, "message", [])
                if hasattr(event, "message_obj")
                else []
            )
        head = self._first_plain_head_text(chain)
        
        at_me = False
        try:
            self_id = event.get_self_id()
            at_me = self._chain_has_at_me(chain, self_id)
        except Exception:
            at_me = False

        if isinstance(head, str) and head.strip():
            hs = head.strip()
            if re.match(r"^\s*/\s*zssm(\s|$)", hs, re.I):
                return
            if at_me and re.match(r"^zssm(\s|$)", hs, re.I):
                return
            if self._is_zssm_trigger(hs):
                async for r in self.zssm(event):
                    yield r
                return
        
        try:
            text = event.get_message_str()
        except Exception:
            text = getattr(event, "message_str", "") or ""

        if isinstance(text, str) and text.strip():
            t = text.strip()
            if re.match(r"^\s*/\s*zssm(\s|$)", t, re.I):
                return
            if at_me and re.match(r"^zssm(\s|$)", t, re.I):
                return
            if self._is_zssm_trigger(t):
                async for r in self.zssm(event):
                    yield r
