#!/usr/bin/env python3
"""Robustness checks (stdlib only).

mixed:  one long JSON-schema constrained generation, with short requests and a
        forced tool call admitted while it decodes (the pattern that crashed
        Aevonix's async-scheduled vLLM deployment); repeated --rounds times.
needle: exact retrieval of three facts hidden in ~N tokens of filler.

usage: robustness.py mixed  [--rounds 3] --out FILE
       robustness.py needle [--tokens 131072] --out FILE
"""
import argparse, json, random, threading, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["mixed", "needle"])
p.add_argument("--base", default="http://127.0.0.1:30000/v1")
p.add_argument("--model", default="mimo-v2.6-pro")
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--tokens", type=int, default=131072)
p.add_argument("--out", required=True)
a = p.parse_args()


def chat(messages, timeout=3600, **kw):
    body = {"model": a.model, "messages": messages, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, **kw}
    req = urllib.request.Request(a.base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r, time.time() - t0


def mixed():
    schema = {"type": "object", "properties": {"items": {"type": "array", "minItems": 300, "maxItems": 300,
              "items": {"type": "integer"}}}, "required": ["items"], "additionalProperties": False}
    tools = [{"type": "function", "function": {"name": "lookup_order", "description": "Look up an order by id",
              "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]
    rounds = []
    for rnd in range(a.rounds):
        out = {}

        def long_job():
            r, s = chat([{"role": "user", "content": "Return JSON with key items: the integers 1 to 300 in order."}],
                        max_tokens=3000, response_format={"type": "json_schema", "json_schema": {"name": "seq", "schema": schema}})
            items = json.loads(r["choices"][0]["message"]["content"])["items"]
            out["long"] = {"ok": items == list(range(1, 301)), "seconds": round(s, 2), "tokens": r["usage"]["completion_tokens"]}

        t = threading.Thread(target=long_job); t.start(); time.sleep(3)
        shorts = []
        for i, (x, y) in enumerate([(17, 25), (123, 456), (9, 91)]):
            r, s = chat([{"role": "user", "content": f"What is {x} + {y}? Reply with the number only."}], max_tokens=16)
            shorts.append({"ok": str(x + y) in r["choices"][0]["message"]["content"], "seconds": round(s, 2)})
            time.sleep(1)
        r, s = chat([{"role": "user", "content": "Where is my order A-7731?"}], tools=tools,
                    tool_choice={"type": "function", "function": {"name": "lookup_order"}}, max_tokens=128)
        tc = r["choices"][0]["message"].get("tool_calls") or []
        tool_ok = bool(tc) and "A-7731" in tc[0]["function"]["arguments"]
        t.join()
        rounds.append({"round": rnd, "long": out.get("long"), "short": shorts, "tool_call_ok": tool_ok})
        print(json.dumps(rounds[-1]), flush=True)
    ok = all(r["long"] and r["long"]["ok"] and r["tool_call_ok"] and all(s["ok"] for s in r["short"]) for r in rounds)
    return {"ok": ok, "rounds": rounds}


def needle():
    rnd = random.Random(a.tokens)
    words = "amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid".split()
    facts = {"vault code": str(rnd.randint(100000, 999999)), "courier name": "Tamsin Okonkwo-Reyes",
             "ship name": "Halcyon Ferrule"}
    n_words = int(a.tokens / 4.6)  # "word123" filler is ~4.6 tokens per word
    body = [f"{rnd.choice(words)}{rnd.randint(0, 999)}" for _ in range(n_words)]
    for i, (k, v) in enumerate(facts.items()):
        pos = int(n_words * (0.15 + 0.35 * i))
        body.insert(pos, f". IMPORTANT FACT: the {k} is {v}. ")
    prompt = (" ".join(body) + "\n\nFrom the text above, what are the vault code, the courier name and the ship name? "
              "Answer as three lines: vault code: ..., courier name: ..., ship name: ...")
    r, s = chat([{"role": "user", "content": prompt}], max_tokens=128)
    ans = r["choices"][0]["message"]["content"]
    hits = {k: v in ans for k, v in facts.items()}
    return {"ok": all(hits.values()), "prompt_tokens": r["usage"]["prompt_tokens"], "seconds": round(s, 2),
            "hits": hits, "answer": ans}


res = mixed() if a.mode == "mixed" else needle()
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rounds"}))
