import os, json, asyncio, time, re
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import urllib.request, urllib.error
from openai import OpenAI

app = FastAPI()

# ── 静态文件 ──────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=Path(__file__).parent), name="static")


# ── 请求体 ────────────────────────────────────────────────
class GenRequest(BaseModel):
    episode_url: str


# ── 首页 ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text()


# ── 主接口：SSE 流式返回进度 ──────────────────────────────
@app.post("/generate")
async def generate(req: GenRequest):
    async def stream():
        def send(event: str, data: str):
            return f"event: {event}\ndata: {data}\n\n"

        dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not dashscope_key:
            yield send("error", "服务器未配置 DASHSCOPE_API_KEY，请联系管理员")
            return

        try:
            # 1. 解析小宇宙链接
            yield send("progress", "正在解析小宇宙链接…")
            meta, shownotes, transcript = await asyncio.get_event_loop().run_in_executor(
                None, fetch_episode, req.episode_url, dashscope_key
            )
            yield send("progress", f"✅ 获取到：{meta['title']}")

            if not transcript:
                yield send("progress", "音频转录中，可能需要 2-5 分钟…")
                transcript = await asyncio.get_event_loop().run_in_executor(
                    None, transcribe_audio, meta["audio_url"], dashscope_key
                )
                yield send("progress", "✅ 转录完成")

            # 2. 用 DeepSeek 生成笔记
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not deepseek_key:
                yield send("error", "服务器未配置 DEEPSEEK_API_KEY，请联系管理员")
                return

            yield send("progress", "正在用 AI 生成结构化笔记…")
            client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
            prompt = build_prompt(meta, shownotes, transcript)

            stream_resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                stream=True,
            )
            for chunk in stream_resp:
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield send("chunk", text)

            yield send("done", json.dumps({"title": meta["title"]}, ensure_ascii=False))

        except Exception as e:
            yield send("error", str(e))

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── 工具函数 ──────────────────────────────────────────────

def _http(url: str, headers: dict = None, data=None, timeout=30):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_episode(episode_url: str, dashscope_key: str):
    """解析小宇宙单集页面，返回 (meta, shownotes, official_transcript)"""
    import html, re

    if not episode_url.startswith("http"):
        episode_url = f"https://www.xiaoyuzhoufm.com/episode/{episode_url}"

    req = urllib.request.Request(episode_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        page = r.read().decode("utf-8", errors="replace")

    # 提取 __NEXT_DATA__
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page, re.S)
    nd = json.loads(m.group(1)) if m else {}

    ep = {}
    try:
        pp = nd["props"]["pageProps"]
        for k in ("episode", "data", "detail"):
            v = pp.get(k)
            if isinstance(v, dict) and ("enclosure" in v or "title" in v):
                ep = v
                break
        if not ep:
            stack = [pp]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    if "enclosure" in cur and "title" in cur:
                        ep = cur
                        break
                    stack.extend(cur.values())
    except Exception:
        pass

    audio_url = ep.get("enclosure", {}).get("url", "") or ep.get("audio_url", "")
    title = ep.get("title", "未知标题")
    podcast_title = ep.get("podcast", {}).get("title", "") or ep.get("podcast_title", "")
    pub_date = ep.get("pubDate", "") or ep.get("pub_date", "")
    duration_sec = ep.get("enclosure", {}).get("duration", 0) or ep.get("duration", 0)

    # 官方转录
    transcript = ""
    for k in ("transcript", "subtitles", "captions", "textTrack"):
        v = ep.get(k)
        if isinstance(v, str) and len(v) > 200:
            transcript = v
            break
        if isinstance(v, list):
            t = "\n".join(x.get("text", "") for x in v if isinstance(x, dict))
            if len(t) > 200:
                transcript = t
                break

    # shownotes
    raw_desc = ep.get("description", "") or ep.get("shownotes", "")

    def html_to_md(s):
        if not s:
            return ""
        s = re.sub(r'<br\s*/?>', '\n', s, flags=re.I)
        s = re.sub(r'</p>', '\n\n', s, flags=re.I)
        s = re.sub(r'<p[^>]*>', '', s, flags=re.I)
        s = re.sub(r'<li[^>]*>', '- ', s, flags=re.I)
        s = re.sub(r'<a[^>]*?href=["\'](.*?)["\'][^>]*?>(.*?)</a>', r'[\2](\1)', s, flags=re.I|re.S)
        s = re.sub(r'<[^>]+>', '', s)
        s = html.unescape(s)
        return re.sub(r'\n{3,}', '\n\n', s).strip()

    shownotes = html_to_md(raw_desc)
    meta = {
        "title": title,
        "podcast_title": podcast_title,
        "pub_date": pub_date,
        "duration_sec": int(duration_sec),
        "audio_url": audio_url,
        "url": episode_url,
    }
    return meta, shownotes, transcript


def transcribe_audio(audio_url: str, dashscope_key: str) -> str:
    """用阿里云百炼 paraformer 转录音频，返回纯文本"""
    api_url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
    headers = {
        "Authorization": f"Bearer {dashscope_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    body = json.dumps({
        "model": "paraformer-v2",
        "input": {"file_urls": [audio_url]},
        "parameters": {"language_hints": ["zh"], "disfluency_removal": True},
    }).encode()

    resp = _http(api_url, headers=headers, data=body)
    task_id = resp.get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"提交转录失败: {resp}")

    # 轮询
    poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    poll_headers = {"Authorization": f"Bearer {dashscope_key}"}
    for _ in range(120):
        time.sleep(5)
        r = _http(poll_url, headers=poll_headers)
        status = r.get("output", {}).get("task_status", "")
        if status == "SUCCEEDED":
            results = r.get("output", {}).get("results", [])
            if results:
                txt_url = results[0].get("transcription_url", "")
                if txt_url:
                    with urllib.request.urlopen(txt_url, timeout=30) as f:
                        raw = json.loads(f.read())
                    texts = []
                    for sent in raw.get("transcripts", [{}])[0].get("sentences", []):
                        texts.append(sent.get("text", ""))
                    return "\n".join(texts)
            break
        if status in ("FAILED", "CANCELED"):
            raise RuntimeError(f"转录任务失败: {r}")

    raise RuntimeError("转录超时，请稍后重试")


def build_prompt(meta: dict, shownotes: str, transcript: str) -> str:
    duration_min = meta["duration_sec"] // 60
    return f"""---
播客：{meta['podcast_title']}
单集：{meta['title']}
发布日期：{meta['pub_date']}
时长：约 {duration_min} 分钟
链接：{meta['url']}
---

**Shownotes**
{shownotes[:3000] if shownotes else '（无）'}

**播客文字稿**
{transcript[:12000] if transcript else '（无转录）'}

---

# 角色
你是一位专业的播客内容分析师，擅长从对话中提炼核心思想、发现深层洞见，
并以结构化方式呈现知识精华。

# 任务
请阅读以上播客文字稿，生成一份结构化的归档笔记。
笔记的核心价值在于**思想提炼与观点启发**，而非简单复述内容。

# 输出结构

## 一、核心摘要
用 3–5 句话概括本期播客最值得记住的核心主张。
要求：站在"听完这期，最重要的收获是什么"的角度来写，而非描述话题本身。

## 二、关键观点
列出 5–10 个最具启发性的观点。每条格式：
**观点**：[一句话提炼，不超过 30 字]
说明：[1–2 句展开，解释为何有洞见或值得记录]

## 三、时间线摘要
按内容推进顺序将播客划分为若干个阶段，每阶段用 1–2 句话概括讨论内容与结论。若文字稿含时间戳则标注；否则用"开篇 / 中段 / 收尾"等自然节点划分。
格式：【阶段标题】— 内容摘要

## 四、金句摘录
从原文摘录 5–8 句最能引发思考的句子（直接引用原文），每句附一行注释说明其价值或背景。
格式：
> 原文金句
注：[注释]

## 五、问答整理
- 若对话中有明确问答：整理为 Q&A，提炼问题核心与回答要点
- 若无明确问答：从内容中提炼 3–5 个"这期播客实际在回答的核心问题"并给出答案摘要
格式：
**Q**：[问题]
**A**：[回答要点，2–4 句]

# 注意事项
- 所有内容基于原文，不添加外部知识
- 观点优先选择：反直觉的、跨领域的、有实践价值的
- 金句标准：读完让人想停下来思考，而不只是"说得好听"
- 文字稿过长时，优先保证摘要和关键观点质量，时间线可适当合并阶段
- 输出语言与文字稿语言一致
- 直接输出笔记正文，不要有任何开场白、说明文字或"好的"等前缀"""
