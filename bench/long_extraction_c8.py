"""Aevonix's 4 long extraction tasks at concurrency 8 (each task twice per wave),
cold cache (flush + unique cache_salt), end to end. Aggregate = tokens / wave wall time.
REPO_DIR is a checkout of https://github.com/Aevonix/mimo-2.6-dgx-spark
(its scripts/benchmark.py and fixtures/). Usage: long_extraction_c8.py BASE_URL REPO_DIR WAVES"""
import hashlib, importlib.util, json, sys, time, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
base, repo, waves = sys.argv[1], sys.argv[2], int(sys.argv[3])
spec = importlib.util.spec_from_file_location("b", f"{repo}/scripts/benchmark.py"); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
oracles = json.load(open(f"{repo}/fixtures/oracles.json"))
rows = [json.loads(l) for l in open(f"{repo}/fixtures/decode.jsonl")]
def one(row):
    prompt = row["conversations"][0]["value"]
    expected = oracles[hashlib.sha256(json.dumps(prompt).encode()).hexdigest()]["expected"]
    body = {"model": "mimo-v2.6-pro", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "seed": 20260922, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False},
            "stream": True, "stream_options": {"include_usage": True}, "cache_salt": "mimo-public-" + uuid.uuid4().hex}
    r = b.run_case(base, body, 900)
    ok = r["finish_reason"] == "stop" and b.exact(expected, b.parsed_json(r["content"]))
    return row["id"], ok, r["usage"]["completion_tokens"], r["elapsed_s"], r["first_model_output_s"]
tot_tok = tot_wall = 0; passed = n = 0; per_stream = []
for w in range(waves):
    urllib.request.urlopen(urllib.request.Request(base.replace("/v1", "") + "/flush_cache", data=b"", method="POST")).read()
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(one, rows * 2))
    wall = time.time() - t0; tok = sum(r[2] for r in res)
    tot_tok += tok; tot_wall += wall; passed += sum(r[1] for r in res); n += len(res)
    per_stream += [r[2] / r[3] for r in res]
    print(json.dumps({"wave": w, "tokens": tok, "wall_s": round(wall, 2), "aggregate_tok_s": round(tok / wall, 2),
                      "passed": sum(r[1] for r in res), "rows": [(r[0], r[1], r[2], round(r[3], 2), round(r[4], 2)) for r in res]}), flush=True)
print(json.dumps({"concurrency": 8, "requests": n, "passed": passed, "tokens": tot_tok, "wall_s": round(tot_wall, 2),
                  "aggregate_tok_s": round(tot_tok / tot_wall, 2), "mean_per_stream_tok_s": round(sum(per_stream) / len(per_stream), 2)}))
