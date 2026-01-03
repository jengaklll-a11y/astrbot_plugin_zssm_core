from __future__ import annotations

from typing import List, Optional, Any, Dict, Tuple

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse, unquote

try:
    import aiohttp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    aiohttp = None

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp

from .message_utils import ob_data


def _safe_subprocess_run(cmd: List[str]) -> subprocess.CompletedProcess:
    if not isinstance(cmd, list) or not cmd:
        raise ValueError("cmd must be a non-empty list")
    if not all(isinstance(x, str) for x in cmd):
        raise TypeError("cmd items must be str")
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        check=False,
        shell=False,
    )


def extract_videos_from_chain(chain: List[object]) -> List[str]:
    """从消息链中递归提取视频相关 URL / 路径。"""
    videos: List[str] = []
    if not isinstance(chain, list):
        return videos

    def _looks_like_video(name_or_url: str) -> bool:
        if not isinstance(name_or_url, str) or not name_or_url:
            return False
        s = name_or_url.lower()
        return any(
            s.endswith(ext)
            for ext in (
                ".mp4",
                ".mov",
                ".m4v",
                ".avi",
                ".webm",
                ".mkv",
                ".flv",
                ".wmv",
                ".ts",
                ".mpeg",
                ".mpg",
                ".3gp",
                ".gif",
            )
        )

    for seg in chain:
        try:
            if hasattr(Comp, "Video") and isinstance(seg, getattr(Comp, "Video")):
                f = getattr(seg, "file", None)
                u = getattr(seg, "url", None)
                if isinstance(u, str) and u:
                    videos.append(u)
                elif isinstance(f, str) and f:
                    videos.append(f)
            elif hasattr(Comp, "File") and isinstance(seg, getattr(Comp, "File")):
                u = getattr(seg, "url", None)
                f = getattr(seg, "file", None)
                n = getattr(seg, "name", None)
                cand = None
                if isinstance(u, str) and u and _looks_like_video(u):
                    cand = u
                elif (
                    isinstance(f, str)
                    and f
                    and (_looks_like_video(f) or os.path.isabs(f))
                ):
                    cand = f
                elif (
                    isinstance(n, str)
                    and n
                    and _looks_like_video(n)
                    and isinstance(f, str)
                    and f
                ):
                    cand = f
                if isinstance(cand, str) and cand:
                    videos.append(cand)
            elif hasattr(Comp, "Node") and isinstance(seg, getattr(Comp, "Node")):
                content = getattr(seg, "content", None)
                if isinstance(content, list):
                    videos.extend(extract_videos_from_chain(content))
            elif hasattr(Comp, "Nodes") and isinstance(seg, getattr(Comp, "Nodes")):
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            videos.extend(extract_videos_from_chain(c))
            elif hasattr(Comp, "Forward") and isinstance(seg, getattr(Comp, "Forward")):
                nodes = getattr(seg, "nodes", None) or getattr(seg, "content", None)
                if isinstance(nodes, list):
                    for node in nodes:
                        c = getattr(node, "content", None)
                        if isinstance(c, list):
                            videos.extend(extract_videos_from_chain(c))
        except Exception:
            continue
    return videos


def is_http_url(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))


def is_abs_file(s: Optional[str]) -> bool:
    return isinstance(s, str) and os.path.isabs(s)


def is_napcat(event: AstrMessageEvent) -> bool:
    try:
        if not (hasattr(event, "bot") and hasattr(event.bot, "api")):
            return False
        api = getattr(event.bot, "api", None)
        if api is None or not hasattr(api, "call_action"):
            return False
        return True
    except Exception:
        return False


async def napcat_resolve_file_url(
    event: AstrMessageEvent, file_id: str
) -> Optional[str]:
    if not (isinstance(file_id, str) and file_id):
        return None
    if not is_napcat(event):
        return None
    try:
        gid = event.get_group_id()
    except Exception:
        gid = None

    group_id_param: Any = gid
    try:
        if isinstance(gid, str) and gid.isdigit():
            group_id_param = int(gid)
        elif isinstance(gid, int):
            group_id_param = gid
    except Exception:
        group_id_param = gid

    def _stem_if_needed(s: str) -> Optional[str]:
        try:
            base, ext = os.path.splitext(s)
            if ext and ext.lower() in (
                ".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv", ".flv", ".wmv", ".ts", ".mpeg", ".mpg", ".3gp",
                ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif",
            ):
                if base and base != s:
                    return base
        except Exception:
            pass
        return None

    candidates: List[str] = [file_id]
    stem = _stem_if_needed(file_id)
    if isinstance(stem, str) and stem and stem not in candidates:
        candidates.append(stem)

    actions: List[Dict[str, Any]] = []
    for fid in candidates:
        actions.append({"action": "get_file", "params": {"file_id": fid}})
        actions.append({"action": "get_file", "params": {"file": fid}})
        actions.append({"action": "get_image", "params": {"file": fid}})
        actions.append({"action": "get_image", "params": {"file_id": fid}})
        actions.append({"action": "get_image", "params": {"id": fid}})
        actions.append({"action": "get_image", "params": {"image": fid}})

    if group_id_param:
        for fid in candidates:
            actions.append(
                {
                    "action": "get_group_file_url",
                    "params": {"group_id": group_id_param, "file_id": fid},
                }
            )
    for fid in candidates:
        actions.append({"action": "get_private_file_url", "params": {"file_id": fid}})

    for item in actions:
        action = item["action"]
        params = item["params"]
        try:
            ret = await event.bot.api.call_action(action, **params)
            data: Optional[Dict[str, Any]]
            if isinstance(ret, dict):
                d = ret.get("data")
                data = d if isinstance(d, dict) else ret
            else:
                data = None
            url = data.get("url") if isinstance(data, dict) else None
            if isinstance(url, str) and url:
                return url
            f = data.get("file") if isinstance(data, dict) else None
            if isinstance(f, str) and f:
                lf = f.lower()
                if lf.startswith("base64://") or lf.startswith("data:image/"):
                    return f
                if lf.startswith("file://"):
                    try:
                        fp = f[7:]
                        if fp.startswith("/") and len(fp) > 3 and fp[2] == ":":
                            fp = fp[1:]
                        if fp and os.path.exists(fp):
                            fp = os.path.abspath(fp)
                            return fp
                    except Exception:
                        pass
                try:
                    if os.path.isabs(f) and os.path.exists(f):
                        return f
                    if os.path.exists(f):
                        fp = os.path.abspath(f)
                        return fp
                except Exception:
                    pass
        except Exception:
            continue
    return None


def resolve_ffmpeg(config_path: str, default_path: str) -> Optional[str]:
    path = config_path or default_path
    if path and shutil.which(path):
        return shutil.which(path)
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    return None


def resolve_ffprobe(ffmpeg_path: Optional[str]) -> Optional[str]:
    sys_ffprobe = shutil.which("ffprobe")
    if sys_ffprobe:
        return sys_ffprobe
    if ffmpeg_path:
        cand = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
        if os.path.exists(cand):
            return cand
    return None


async def sample_frames_with_ffmpeg(
    ffmpeg_path: str,
    video_path: str,
    interval_sec: int,
    count_limit: int,
) -> List[str]:
    out_dir = tempfile.mkdtemp(prefix="zssm_frames_")
    out_tpl = os.path.join(out_dir, "frame_%03d.jpg")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps=1/{max(1, interval_sec)}",
        "-frames:v",
        str(max(1, count_limit)),
        "-qscale:v",
        "2",
        out_tpl,
    ]
    loop = asyncio.get_running_loop()

    def _run():
        return _safe_subprocess_run(cmd)

    res = await loop.run_in_executor(None, _run)
    if res.returncode != 0:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
        raise RuntimeError("ffmpeg sample frames failed")

    frames: List[str] = []
    try:
        for name in sorted(os.listdir(out_dir)):
            if name.lower().endswith(".jpg"):
                frames.append(os.path.join(out_dir, name))
    except Exception:
        pass

    if not frames:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
        raise RuntimeError("no frames generated")
    return frames


async def sample_frames_equidistant(
    ffmpeg_path: str,
    video_path: str,
    duration_sec: float,
    count_limit: int,
) -> List[str]:
    N = max(1, int(count_limit))
    out_dir = tempfile.mkdtemp(prefix="zssm_frames_")
    loop = asyncio.get_running_loop()
    frames: List[str] = []
    times: List[float] = []
    try:
        total = max(0.0, float(duration_sec))
        for i in range(1, N + 1):
            t = (i / (N + 1.0)) * total
            times.append(t)
        for idx, t in enumerate(times, start=1):
            out_path = os.path.join(out_dir, f"frame_{idx:03d}.jpg")
            cmd = [
                ffmpeg_path,
                "-y",
                "-ss",
                f"{max(0.0, t):.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-qscale:v",
                "2",
                out_path,
            ]

            def _run_one(cmd=cmd):
                return _safe_subprocess_run(cmd)

            res = await loop.run_in_executor(None, _run_one)
            if res.returncode == 0 and os.path.exists(out_path):
                frames.append(out_path)
    except Exception as e:
        logger.error("zssm_explain: equidistant sampler error: %s", e)
    if not frames:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
        raise RuntimeError("no frames generated by equidistant sampler")
    return frames


async def download_video_to_temp(
    url: str, size_mb_limit: int, headers: Optional[Dict[str, str]] = None
) -> Optional[str]:
    def _safe_ext_from_url(u: str) -> str:
        try:
            path = urlparse(u).path
            base = os.path.basename(unquote(path))
            ext = os.path.splitext(base)[1]
            if isinstance(ext, str):
                ext = ext[:8]
            if not ext or not re.match(r"^\.[A-Za-z0-9]{1,6}$", ext):
                lower = base.lower()
                for cand in (
                    ".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv", ".flv", ".wmv",
                ):
                    if lower.endswith(cand):
                        return cand
                return ".bin"
            return ext
        except Exception:
            return ".bin"

    ext = _safe_ext_from_url(url)
    tmp = tempfile.NamedTemporaryFile(prefix="zssm_video_", suffix=ext, delete=False)
    tmp_path = tmp.name
    tmp.close()
    max_bytes = size_mb_limit * 1024 * 1024
    if aiohttp is not None:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=20, headers=headers or {}) as resp:
                    if resp.status != 200:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        return None
                    cl = resp.headers.get("Content-Length")
                    if cl and cl.isdigit() and int(cl) > max_bytes:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        return None
                    total = 0
                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > max_bytes:
                                try:
                                    f.close()
                                except Exception:
                                    pass
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass
                                return None
                            f.write(chunk)
            return tmp_path if os.path.exists(tmp_path) else None
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return None
    try:
        import urllib.request

        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=20) as r, open(tmp_path, "wb") as f:
            total = 0
            while True:
                chunk = r.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    try:
                        f.close()
                    except Exception:
                        pass
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    return None
                f.write(chunk)
        return tmp_path if os.path.exists(tmp_path) else None
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None


def probe_duration_sec(ffprobe_path: Optional[str], video_path: str) -> Optional[float]:
    if not ffprobe_path:
        return None
    candidates: List[float] = []
    try:
        cmd1 = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            video_path,
        ]
        res1 = _safe_subprocess_run(cmd1)
        if res1.returncode == 0:
            try:
                data1 = json.loads(res1.stdout.decode("utf-8", errors="ignore") or "{}")
            except Exception:
                data1 = {}
            if isinstance(data1, dict):
                fmt = data1.get("format")
                if isinstance(fmt, dict):
                    d = fmt.get("duration")
                    try:
                        dur = float(d)
                        if dur and dur > 0:
                            candidates.append(dur)
                    except Exception:
                        pass

        cmd2 = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration,nb_frames,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            video_path,
        ]
        res2 = _safe_subprocess_run(cmd2)
        if res2.returncode == 0:
            try:
                data2 = json.loads(res2.stdout.decode("utf-8", errors="ignore") or "{}")
            except Exception:
                data2 = {}
            stream = None
            if isinstance(data2, dict):
                streams = data2.get("streams")
                if isinstance(streams, list) and streams:
                    s0 = streams[0]
                    if isinstance(s0, dict):
                        stream = s0
            if isinstance(stream, dict):
                d = stream.get("duration")
                try:
                    dur = float(d)
                    if dur and dur > 0:
                        candidates.append(dur)
                except Exception:
                    pass
                fps_txt = (
                    stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
                )
                try:
                    num, den = fps_txt.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 0.0
                except Exception:
                    fps = 0.0
                try:
                    nb_frames = stream.get("nb_frames")
                    nb = (
                        int(nb_frames)
                        if nb_frames is not None and str(nb_frames).isdigit()
                        else 0
                    )
                except Exception:
                    nb = 0
                if fps > 0 and nb > 0:
                    cand = nb / fps
                    if cand > 0:
                        candidates.append(cand)
    except Exception as e:
        logger.warning("zssm_explain: ffprobe duration failed: %s", e)
    if not candidates:
        return None
    c_sorted = sorted(set(candidates))
    mid = len(c_sorted) // 2
    chosen = c_sorted[mid]
    return chosen


async def extract_forward_video_keyframes(
    event: AstrMessageEvent,
    video_sources: List[str],
    *,
    enabled: bool,
    max_count: int,
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    max_mb: int,
    max_sec: int,
    timeout_sec: int,
) -> Tuple[List[str], List[str]]:
    if not enabled:
        return ([], [])
    if not video_sources:
        return ([], [])
    if not ffmpeg_path:
        return ([], [])
    try:
        max_count = int(max_count)
    except Exception:
        max_count = 0
    if max_count <= 0:
        return ([], [])

    uniq_sources: List[str] = []
    seen = set()
    for s in video_sources:
        if isinstance(s, str) and s and s not in seen:
            seen.add(s)
            uniq_sources.append(s)

    frames: List[str] = []
    cleanup: List[str] = []

    for src in uniq_sources[:max_count]:
        local_path = None
        downloaded_tmp = False
        try:
            resolved_src = src
            if (
                isinstance(resolved_src, str)
                and (not is_http_url(resolved_src))
                and (not is_abs_file(resolved_src))
            ):
                try:
                    resolved = await napcat_resolve_file_url(event, resolved_src)
                except Exception:
                    resolved = None
                if isinstance(resolved, str) and resolved:
                    resolved_src = resolved

            if isinstance(resolved_src, str) and is_http_url(resolved_src):
                try:
                    local_path = await asyncio.wait_for(
                        download_video_to_temp(resolved_src, max_mb),
                        timeout=max(2, int(timeout_sec)),
                    )
                except Exception as e:
                    logger.warning(
                        "zssm_explain: forward video download timeout/failed: %s", e
                    )
                if local_path:
                    downloaded_tmp = True
            elif (
                isinstance(resolved_src, str)
                and is_abs_file(resolved_src)
                and os.path.exists(resolved_src)
            ):
                local_path = resolved_src

            if not local_path:
                continue

            dur = probe_duration_sec(ffprobe_path, local_path) if ffprobe_path else None
            if isinstance(dur, (int, float)) and dur > max_sec:
                continue

            try:
                if isinstance(dur, (int, float)) and dur > 0:
                    sampled = await sample_frames_equidistant(
                        ffmpeg_path, local_path, float(dur), 1
                    )
                else:
                    sampled = await sample_frames_with_ffmpeg(
                        ffmpeg_path, local_path, max(1, max_sec), 1
                    )
            except Exception:
                sampled = []

            if sampled:
                frames.append(sampled[0])
                try:
                    cleanup.append(os.path.dirname(sampled[0]))
                except Exception:
                    pass
        finally:
            if downloaded_tmp and isinstance(local_path, str) and local_path:
                cleanup.append(local_path)
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except Exception:
                    pass

    uniq_cleanup: List[str] = []
    seen2 = set()
    for p in cleanup:
        if isinstance(p, str) and p and p not in seen2:
            seen2.add(p)
            uniq_cleanup.append(p)

    return (frames, uniq_cleanup)
