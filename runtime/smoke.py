#!/usr/bin/env python3
"""Functional smoke test + single-stream decode speed for the MiMo endpoint.

Stdlib only. Usage: smoke.py [--base http://127.0.0.1:30000/v1] [--out results.json]
"""
import argparse, base64, json, struct, sys, time, urllib.request, zlib

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:30000/v1")
p.add_argument("--model", default="mimo-v2.6-pro")
p.add_argument("--out")
a = p.parse_args()


def post(path, body, stream=False):
    req = urllib.request.Request(a.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900) if stream else json.load(urllib.request.urlopen(req, timeout=900))


def chat(messages, **kw):
    body = {"model": a.model, "messages": messages, "temperature": 0, "max_tokens": 512, **kw}
    return post("/chat/completions", body)["choices"][0]


def png(w, h, rgb_fn):
    raw = b"".join(b"\0" + bytes(c for x in range(w) for c in rgb_fn(x, y)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def stream_speed(prompt, max_tokens, thinking=False):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    t0 = time.time(); first = None; usage = None; text = []
    for line in post("/chat/completions", body, stream=True):
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
            if piece:
                first = first or time.time(); text.append(piece)
    end = time.time(); n = usage["completion_tokens"]
    return {"completion_tokens": n, "ttft_s": round(first - t0, 3),
            "decode_tok_s": round((n - 1) / (end - first), 2), "text_head": "".join(text)[:160]}


R = {}
c = chat([{"role": "user", "content": "What is 19 + 23? Answer with just the number."}],
         chat_template_kwargs={"enable_thinking": False})
R["arithmetic"] = {"ok": "42" in c["message"]["content"], "content": c["message"]["content"]}

c = chat([{"role": "user", "content": "How many r letters are in 'strawberry'? Think, then answer."}], max_tokens=2048)
m = c["message"]
R["reasoning"] = {"ok": bool(m.get("reasoning_content")) and "3" in (m.get("content") or ""),
                  "reasoning_chars": len(m.get("reasoning_content") or ""), "content": (m.get("content") or "")[:200]}

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
c = chat([{"role": "user", "content": "What's the weather in Cardiff right now?"}], tools=tools,
         chat_template_kwargs={"enable_thinking": False})
tc = c["message"].get("tool_calls") or []
R["tool_call"] = {"ok": bool(tc) and tc[0]["function"]["name"] == "get_weather" and "Cardiff" in tc[0]["function"]["arguments"],
                  "tool_calls": tc, "finish_reason": c["finish_reason"]}

# Left half red, right half blue.
img = base64.b64encode(png(256, 128, lambda x, y: (220, 20, 20) if x < 128 else (20, 40, 220))).decode()
c = chat([{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
    {"type": "text", "text": "This image has two halves. What colour is the left half and what colour is the right half? Answer briefly."}]}],
    chat_template_kwargs={"enable_thinking": False})
t = c["message"]["content"].lower()
R["image"] = {"ok": "red" in t and "blue" in t and t.find("red") < t.find("blue"), "content": c["message"]["content"]}

R["speed_code"] = stream_speed("Write a Python function implementing an LRU cache class with get/put, with type hints and docstrings. Code only.", 256)
R["speed_prose"] = stream_speed("Write a short story about a lighthouse keeper who finds a message in a bottle.", 256)

print(json.dumps(R, indent=2))
if a.out:
    json.dump(R, open(a.out, "w"), indent=2)
sys.exit(0 if all(v.get("ok", True) for v in R.values()) else 1)
