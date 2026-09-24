"""OpenLux OpenAI-compatible client.

Official Quick Start:
  pip install openai
  client = OpenAI(api_key=..., base_url="https://api.openlux.ai/v1")
  client.chat.completions.create(model=..., messages=[...])

Docs: https://api.openlux.ai/pricing
Base URL must include /v1.

Env:
  OPENLUX_API_KEY   required
  OPENLUX_BASE_URL  default https://api.openlux.ai/v1
  OPENLUX_MODEL     default gpt-5.6-sol  (override with console model id)
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

DEFAULT_BASE_URL = os.environ.get("OPENLUX_BASE_URL", "https://api.openlux.ai/v1").rstrip("/")
DEFAULT_MODEL = os.environ.get("OPENLUX_MODEL", "gpt-5.6-sol")


def openlux_client(*, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0):
    """Create OpenAI SDK client pointed at OpenLux."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("pip install openai") from exc

    key = api_key or os.environ.get("OPENLUX_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set OPENLUX_API_KEY (console: https://api.openlux.ai/pricing)"
        )
    url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    return OpenAI(api_key=key, base_url=url, timeout=timeout)


def encode_image_data_url(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    suffix = p.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def chat_text(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def multimodal_messages(
    *,
    system: str,
    user_text: str,
    image_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if image_path:
        url = encode_image_data_url(image_path)
        if url:
            user_content.insert(0, {"type": "image_url", "image_url": {"url": url}})
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    return messages


def extract_json_obj(text: str) -> Any:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pat in (r"```json\s*(.*?)\s*```", r"(\{.*\}|\[.*\])"):
        m = re.search(pat, text, re.DOTALL)
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except Exception:
            continue
    return None
