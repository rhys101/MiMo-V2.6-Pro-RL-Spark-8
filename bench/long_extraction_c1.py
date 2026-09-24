"""Re-run Aevonix's 4 long extraction tasks with their exact request bodies and
their run_case() timing (client-side, request start -> end of stream), flushing the
server's prefix cache before every request. REPO_DIR is a checkout of https://github.com/Aevonix/mimo-2.6-dgx-spark
(its scripts/benchmark.py and fixtures/). Usage: long_extraction_c1.py BASE_URL REPO_DIR REPEATS"""
import hashlib, importlib.util, json, sys, urllib.request, uuid
base, repo, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
spec = importlib.util.spec_from_file_location("b", f"{repo}/scripts/benchmark.py"); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
oracles = json.load(open(f"{repo}/fixtures/oracles.json"))
rows = [json.loads(l) for l in open(f"{repo}/fixtures/decode.jsonl")]
tot_tok = tot_s = 0; per = []
for rep in range(reps):
    for row in rows:
        prompt = row["conversations"][0]["value"]
        expected = oracles[hashlib.sha256(json.dumps(prompt).encode()).hexdigest()]["expected"]
        body = {"model": "mimo-v2.6-pro", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "seed": 20260922, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False},
                "stream": True, "stream_options": {"include_usage": True}, "cache_salt": "mimo-public-" + uuid.uuid4().hex}
        urllib.request.urlopen(urllib.request.Request(base.replace("/v1", "") + "/flush_cache", data=b"", method="POST")).read()
        r = b.run_case(base, body, 600)
        ok = r["finish_reason"] == "stop" and b.exact(expected, b.parsed_json(r["content"]))
        n = r["usage"]["completion_tokens"]; tot_tok += n; tot_s += r["elapsed_s"]
        per.append((rep, row["id"], ok, n, round(r["elapsed_s"], 2), round(r["first_model_output_s"], 2), r["usage"].get("prompt_tokens")))
        print(per[-1], flush=True)
print(json.dumps({"requests": len(per), "passed": sum(p[2] for p in per), "tokens": tot_tok, "seconds": round(tot_s, 2),
                  "e2e_tok_s": round(tot_tok / tot_s, 2)}))
