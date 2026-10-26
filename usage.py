"""
usage.py — keeps count of Gemini usage so the app can show what's left.

The Gemini API doesn't report remaining quota, so Margin counts its own calls:
  * per-minute limits (requests and tokens) use a rolling 60-second window
  * daily request limits reset at midnight Pacific time (Google's documented reset)

Counts are shared by everyone using this app (they share one API key) and saved to disk,
so they survive page reloads. Usage from other apps on the same Google project isn't seen.
"""

import datetime as dt
import json
import threading
import time
from collections import deque
from pathlib import Path
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")

DEFAULT_LIMITS = {
    "chat_rpd": 1000,     # chat-model requests per day
    "chat_rpm": 15,       # chat-model requests per minute
    "chat_tpm": 250_000,  # chat-model input tokens per minute
    "embed_rpd": 1000,    # embedding requests per day
    "embed_rpm": 100,     # embedding requests per minute
}


def _today():
    return dt.datetime.now(PACIFIC).date().isoformat()


def seconds_to_midnight_pacific():
    now = dt.datetime.now(PACIFIC)
    tomorrow = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((tomorrow - now).total_seconds())


def human_duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60}h {minutes % 60:02d}m"


class Usage:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.minute = {"chat": deque(), "embed": deque()}  # (timestamp, tokens)
        self.state = self._fresh()
        try:
            saved = json.loads(self.path.read_text())
            if saved.get("day") == _today():
                self.state.update(saved)
        except Exception:
            pass

    def _fresh(self):
        return {"day": _today(), "chat_requests": 0, "embed_requests": 0, "input_tokens": 0,
                "output_tokens": 0, "embed_tokens": 0, "questions": 0, "question_requests": 0,
                "last_limit_hit": None}

    def _roll(self):
        if self.state["day"] != _today():
            self.state = self._fresh()
        cutoff = time.time() - 60
        for q in self.minute.values():
            while q and q[0][0] < cutoff:
                q.popleft()

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state))
        except Exception:
            pass

    def record_chat(self, input_tokens, output_tokens):
        with self.lock:
            self._roll()
            self.state["chat_requests"] += 1
            self.state["input_tokens"] += int(input_tokens)
            self.state["output_tokens"] += int(output_tokens)
            self.minute["chat"].append((time.time(), int(input_tokens)))
            self._save()

    def record_embed(self, texts, chars):
        """Google counts each embedded passage against the embedding quota, even inside one batch call."""
        with self.lock:
            self._roll()
            self.state["embed_requests"] += int(texts)
            self.state["embed_tokens"] += int(chars / 4)
            now = time.time()
            for _ in range(int(texts)):
                self.minute["embed"].append((now, int(chars / 4 / max(texts, 1))))
            self._save()

    def embed_left(self, limit):
        with self.lock:
            self._roll()
            return max(0, limit - self.state["embed_requests"])

    def record_question(self, chat_requests):
        with self.lock:
            self._roll()
            self.state["questions"] += 1
            self.state["question_requests"] += chat_requests
            self._save()

    def record_limit_hit(self, kind):
        with self.lock:
            self.state["last_limit_hit"] = {"kind": kind, "at": time.time()}
            self._save()

    def chat_requests_so_far(self):
        with self.lock:
            self._roll()
            return self.state["chat_requests"]

    def snapshot(self, limits=None):
        limits = {**DEFAULT_LIMITS, **(limits or {})}
        with self.lock:
            self._roll()
            s = dict(self.state)
            chat_min, embed_min = list(self.minute["chat"]), list(self.minute["embed"])
        now = time.time()
        per_q = (s["question_requests"] / s["questions"]) if s["questions"] else 3.0
        chat_left = max(0, limits["chat_rpd"] - s["chat_requests"])
        embed_left = max(0, limits["embed_rpd"] - s["embed_requests"])
        minute_reset = int(60 - (now - chat_min[0][0])) + 1 if chat_min else 0
        limit_hit = s.get("last_limit_hit")
        recent_hit = limit_hit and now - limit_hit["at"] < 60
        return {
            "chat_used": s["chat_requests"], "chat_limit": limits["chat_rpd"], "chat_left": chat_left,
            "chat_pct_left": chat_left / max(limits["chat_rpd"], 1),
            "embed_used": s["embed_requests"], "embed_limit": limits["embed_rpd"], "embed_left": embed_left,
            "input_tokens": s["input_tokens"], "output_tokens": s["output_tokens"], "embed_tokens": s["embed_tokens"],
            "tokens_today": s["input_tokens"] + s["output_tokens"] + s["embed_tokens"],
            "minute_requests": len(chat_min), "minute_rpm": limits["chat_rpm"],
            "minute_tokens": sum(t for _, t in chat_min), "minute_tpm": limits["chat_tpm"],
            "minute_embed": len(embed_min), "minute_embed_rpm": limits["embed_rpm"],
            "minute_reset_s": minute_reset,
            "day_reset_s": seconds_to_midnight_pacific(),
            "requests_per_question": round(per_q, 1),
            "questions_left": int(min(chat_left / max(per_q, 1), embed_left)),
            "questions_today": s["questions"],
            "limited_now": bool(recent_hit),
        }
