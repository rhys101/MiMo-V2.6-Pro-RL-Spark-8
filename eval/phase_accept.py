#!/usr/bin/env python3
"""Reasoning vs answer: token share and speculative accept length per phase.

With stream_interval=1, SGLang emits one stream chunk per decode/verify step, so
accept length per phase ~= tokens in that phase / chunks in that phase. Token counts
come from the server tokenizer (/tokenize); reasoning totals are cross-checked
against usage.reasoning_tokens.

usage: phase_accept.py --out FILE [--base http://127.0.0.1:30000]
"""
import argparse, json, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:30000")
p.add_argument("--model", default="mimo-v2.6-pro")
p.add_argument("--out", required=True)
p.add_argument("--set", choices=["easy", "hard"], default="easy")
a = p.parse_args()

TASKS = {
    "cy": ["Esboniwch sut mae ffotosynthesis yn gweithio, mewn tua 200 gair.",
           "Ysgrifennwch stori fer am bysgotwr yn Aberystwyth sy'n dod o hyd i neges mewn potel.",
           "Beth yw manteision ac anfanteision ynni gwynt ar y môr i Gymru?",
           "Ysgrifennwch swyddogaeth Python sy'n gwirio a yw gair yn balindrom, gydag esboniad yn Gymraeg.",
           "Rhowch gyngor i rywun sy'n dechrau dysgu Cymraeg."],
    "fr": ["Expliquez comment fonctionne la photosynthèse, en environ 200 mots.",
           "Écrivez une courte histoire sur un pêcheur breton qui trouve un message dans une bouteille.",
           "Quels sont les avantages et les inconvénients de l'énergie éolienne en mer pour la France ?",
           "Écrivez une fonction Python qui vérifie si un mot est un palindrome, avec une explication en français.",
           "Donnez des conseils à quelqu'un qui commence à apprendre le français."],
    "en": ["Explain how photosynthesis works, in about 200 words.",
           "Write a short story about a fisherman in Cornwall who finds a message in a bottle.",
           "What are the advantages and disadvantages of offshore wind power for the UK?",
           "Write a Python function that checks whether a word is a palindrome, with an explanation.",
           "Give advice to someone starting to learn Spanish."],
}
KINDS = ["explain", "story", "argue", "code", "advice"]
HARD = {
    "cy": ["Mae pum tŷ mewn rhes, pob un â lliw gwahanol. Mae'r tŷ coch yn union i'r chwith o'r tŷ gwyrdd. Mae'r tŷ glas ym mhen y rhes. Nid yw'r tŷ melyn drws nesaf i'r tŷ glas. Mae'r tŷ gwyn yn y canol. Beth yw trefn y tai? Esboniwch eich rhesymu.",
           "Mae trên yn gadael Caerdydd am 09:40 ac yn teithio 212 km ar 80 km/awr, yn aros am 12 munud, yna'n teithio 95 km ar 60 km/awr. Mae ail drên yn gadael 25 munud yn ddiweddarach ar 90 km/awr heb stopio ar yr un llwybr. Pa drên sy'n cyrraedd gyntaf, a faint o funudau yw'r gwahaniaeth?",
           "Mae gan y swyddogaeth Python hon nam: def cyfartaledd(rhestr): cyfanswm = 0\n for i in range(1, len(rhestr)): cyfanswm += rhestr[i]\n return cyfanswm / len(rhestr). Darganfyddwch y nam, esboniwch pam, a rhowch fersiwn gywir sy'n trin rhestr wag.",
           "Rwy'n trefnu cynhadledd undydd i 120 o bobl gyda thri sesiwn gyfochrog, dau egwyl a chinio. Mae gennyf dair ystafell (60, 40 a 30 sedd). Cynlluniwch amserlen sy'n osgoi gorlenwi, ac esboniwch y cyfaddawdau."],
    "fr": ["Cinq maisons sont alignées, chacune d'une couleur différente. La maison rouge est juste à gauche de la maison verte. La maison bleue est à une extrémité. La maison jaune n'est pas à côté de la bleue. La maison blanche est au milieu. Quel est l'ordre des maisons ? Expliquez votre raisonnement.",
           "Un train quitte Lyon à 9 h 40 et parcourt 212 km à 80 km/h, s'arrête 12 minutes, puis parcourt 95 km à 60 km/h. Un second train part 25 minutes plus tard à 90 km/h sans arrêt sur le même trajet. Lequel arrive en premier, et avec combien de minutes d'écart ?",
           "Cette fonction Python contient un bug : def moyenne(liste): total = 0\n for i in range(1, len(liste)): total += liste[i]\n return total / len(liste). Trouvez le bug, expliquez pourquoi, et donnez une version correcte qui gère une liste vide.",
           "J'organise une conférence d'une journée pour 120 personnes avec trois sessions parallèles, deux pauses et un déjeuner. J'ai trois salles (60, 40 et 30 places). Proposez un programme qui évite la surcharge et expliquez les compromis."],
    "en": ["Five houses stand in a row, each a different colour. The red house is immediately left of the green house. The blue house is at one end. The yellow house is not next to the blue one. The white house is in the middle. What is the order of the houses? Explain your reasoning.",
           "A train leaves Leeds at 09:40 and travels 212 km at 80 km/h, stops for 12 minutes, then travels 95 km at 60 km/h. A second train leaves 25 minutes later at 90 km/h without stopping on the same route. Which arrives first, and by how many minutes?",
           "This Python function has a bug: def average(items): total = 0\n for i in range(1, len(items)): total += items[i]\n return total / len(items). Find the bug, explain why, and give a correct version that handles an empty list.",
           "I'm organising a one-day conference for 120 people with three parallel sessions, two breaks and lunch. I have three rooms (60, 40 and 30 seats). Propose a schedule that avoids overcrowding and explain the trade-offs."],
}
HARD_KINDS = ["logic", "math", "debug", "plan"]
if a.set == "hard":
    TASKS, KINDS = HARD, HARD_KINDS


def ntok(text):
    if not text:
        return 0
    req = urllib.request.Request(a.base + "/tokenize", json.dumps({"model": a.model, "prompt": text}).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["count"]


def run(prompt):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 1.0, "top_p": 0.95,
            "max_tokens": 6000, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    ph = {"reasoning": {"text": "", "chunks": 0, "t0": None, "t1": None}, "answer": {"text": "", "chunks": 0, "t0": None, "t1": None}}
    usage = None
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        usage = ev.get("usage") or usage
        for ch in ev.get("choices", []):
            d = ch.get("delta", {}); now = time.time()
            for key, field in (("reasoning", "reasoning_content"), ("answer", "content")):
                piece = d.get(field) or ""
                if piece:
                    s = ph[key]; s["text"] += piece; s["chunks"] += 1
                    s["t0"] = s["t0"] or now; s["t1"] = now
    out = {"completion_tokens": usage["completion_tokens"], "usage_reasoning_tokens": usage.get("reasoning_tokens")}
    for key, s in ph.items():
        n = ntok(s["text"])
        dur = (s["t1"] - s["t0"]) if s["t0"] else 0
        out[key] = {"tokens": n, "chunks": s["chunks"], "accept_len": round(n / s["chunks"], 2) if s["chunks"] else None,
                    "tok_s": round((n - 1) / dur, 1) if dur > 0 and n > 1 else None, "head": s["text"][:120]}
    return out


results = []
for lang, prompts in TASKS.items():
    for kind, prompt in zip(KINDS, prompts):
        r = run(prompt); r.update(lang=lang, kind=kind)
        results.append(r)
        print(json.dumps({k: r[k] for k in ("lang", "kind", "completion_tokens", "usage_reasoning_tokens")}),
              "| reasoning", {k: r["reasoning"][k] for k in ("tokens", "accept_len", "tok_s")},
              "| answer", {k: r["answer"][k] for k in ("tokens", "accept_len", "tok_s")}, flush=True)

summary = {}
for lang in list(TASKS) + ["all"]:
    rs = [r for r in results if lang == "all" or r["lang"] == lang]
    s = {}
    for key in ("reasoning", "answer"):
        t = sum(r[key]["tokens"] for r in rs); c = sum(r[key]["chunks"] for r in rs)
        s[key + "_tokens"] = t; s[key + "_accept_len"] = round(t / c, 2) if c else None
    tot = s["reasoning_tokens"] + s["answer_tokens"]
    s["reasoning_share"] = round(s["reasoning_tokens"] / tot, 3) if tot else None
    summary[lang] = s
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "results": results}, open(a.out, "w"), indent=1, ensure_ascii=False)
