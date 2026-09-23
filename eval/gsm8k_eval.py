#!/usr/bin/env python3
"""GSM8K accuracy A/B check for kernel/backends changes (stdlib only).

Greedy, thinking off, first N test questions, fixed order. The goal is to
detect numerical regressions between builds, not to report a leaderboard score.

usage: gsm8k_eval.py --data gsm8k-test.jsonl --n 200 --out result.json [--base URL] [--concurrency 8]
"""
import argparse, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:30000/v1")
p.add_argument("--model", default="mimo-v2.6-pro")
p.add_argument("--data", required=True)
p.add_argument("--n", type=int, default=200)
p.add_argument("--concurrency", type=int, default=8)
p.add_argument("--thinking", action="store_true")
p.add_argument("--out", required=True)
a = p.parse_args()

PROMPT = ("Solve the following math problem. Show brief working, then give the final answer on the last line "
          "in the form 'Answer: <number>'.\n\n{q}")


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def pred(text):
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text or "")
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text or "")
    return m[-1].replace(",", "").rstrip(".") if m else None


def same(p_, g):
    try:
        return p_ is not None and abs(float(p_) - float(g)) < 1e-6
    except ValueError:
        return False


def ask(item):
    i, row = item
    body = {"model": a.model, "temperature": 0, "max_tokens": 4096 if a.thinking else 1024,
            "messages": [{"role": "user", "content": PROMPT.format(q=row["question"])}],
            "chat_template_kwargs": {"enable_thinking": a.thinking}}
    req = urllib.request.Request(a.base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    text = r["choices"][0]["message"]["content"]
    g, pr = gold(row["answer"]), pred(text)
    return {"i": i, "gold": g, "pred": pr, "ok": same(pr, g), "tokens": r["usage"]["completion_tokens"],
            "seconds": round(time.time() - t0, 2), "text": text}


rows = [json.loads(l) for l in open(a.data)][: a.n]
t0 = time.time()
with ThreadPoolExecutor(a.concurrency) as ex:
    res = sorted(ex.map(ask, enumerate(rows)), key=lambda r: r["i"])
acc = sum(r["ok"] for r in res) / len(res)
summary = {"n": len(res), "accuracy": round(acc, 4), "correct": sum(r["ok"] for r in res),
           "wall_s": round(time.time() - t0, 1), "thinking": a.thinking,
           "mean_tokens": round(sum(r["tokens"] for r in res) / len(res), 1)}
json.dump({"summary": summary, "results": res}, open(a.out, "w"), indent=1)
print(json.dumps(summary))
