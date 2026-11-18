"""Provider-agnostic LLM labelling over plain HTTPS (no SDK needed). Works with a FREE key.
Set ONE of:   GEMINI_API_KEY (free tier at aistudio.google.com)  |  GROQ_API_KEY (free tier)  |  ANTHROPIC_API_KEY
Optional:     LLM_PROVIDER=gemini|groq|anthropic   LLM_MODEL=<model name>   LLM_BATCH=25   LLM_RPM=10
No key -> available() is False and the caller falls back to the local classifier."""
import os, json, time, hashlib, re, urllib.request, urllib.error

DEFAULT_MODEL = {"gemini": "gemini-2.0-flash", "groq": "llama-3.3-70b-versatile", "anthropic": "claude-haiku-4-5"}


def provider():
    p = os.environ.get("LLM_PROVIDER", "").lower()
    if p:
        return p
    for name, var in (("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")):
        if os.environ.get(var):
            return name
    return ""


def available():
    return bool(provider())


def _post(url, headers, body, timeout=90):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call_llm(prompt):
    p = provider()
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL.get(p, ""))
    if p == "gemini":
        key = os.environ["GEMINI_API_KEY"]
        d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}", {},
                  {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
        return d["candidates"][0]["content"]["parts"][0]["text"]
    if p == "groq":
        d = _post("https://api.groq.com/openai/v1/chat/completions", {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
                  {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
        return d["choices"][0]["message"]["content"]
    if p == "anthropic":
        d = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                  {"model": model, "max_tokens": 4000, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
        return d["content"][0]["text"]
    raise RuntimeError("no LLM provider configured")


def build_prompt(leaf_paths, items):
    tree = "\n".join(f"{i}: {' > '.join(p)}" for i, p in enumerate(leaf_paths))
    lines = "\n".join(f"{k}: supplier={s!r} | text={t!r}" for k, (s, t) in enumerate(items))
    return ("You classify purchase-order lines into a company's category tree. Descriptions are short, abbreviated, may have OCR "
            "errors, be in any language, or be uninformative; use the supplier name as a clue. Pick exactly one leaf id per line.\n\n"
            f"CATEGORY LEAVES (id: path):\n{tree}\n\nLINES:\n{lines}\n\n"
            'Answer ONLY with a JSON array, one object per line, in order: [{"i": <line number>, "leaf": <leaf id>, "conf": <0..1>}, ...]')


def _parse(txt, n, n_leaves):
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        arr = json.loads(txt)
    except Exception:
        m = re.search(r"\[.*\]", txt, re.S)
        arr = json.loads(m.group(0)) if m else []
    out = {}
    for o in arr if isinstance(arr, list) else []:
        try:
            i, leaf = int(o["i"]), int(o["leaf"])
            if 0 <= i < n and 0 <= leaf < n_leaves:
                out[i] = (leaf, float(o.get("conf", 0.7)))
        except Exception:
            continue
    return out


def label_items(leaf_paths, items, call_fn=None, batch=None, rpm=None, cache_path=".cache/llm_cache.json", log=print):
    """items: list of (supplier, text). Returns list of (leaf_idx, conf) or None per item. Cached on disk."""
    call_fn = call_fn or call_llm
    batch = batch or int(os.environ.get("LLM_BATCH", 25))
    rpm = rpm or float(os.environ.get("LLM_RPM", 10))
    sig = hashlib.md5(json.dumps(leaf_paths).encode()).hexdigest()[:8]
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}
    ck = lambda it: sig + "|" + hashlib.md5(json.dumps(it).encode()).hexdigest()
    res = [tuple(cache[ck(it)]) if ck(it) in cache else None for it in items]
    todo = [i for i, r in enumerate(res) if r is None]
    log(f"llm: {len(items)} unique items, {len(items) - len(todo)} cached, {len(todo)} to label in ~{-(-len(todo) // batch)} calls")
    last = 0.0
    for b in range(0, len(todo), batch):
        ids = todo[b:b + batch]
        prompt = build_prompt(leaf_paths, [items[i] for i in ids])
        got = {}
        for attempt in range(4):
            wait = 60.0 / rpm - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            try:
                got = _parse(call_fn(prompt), len(ids), len(leaf_paths))
                break
            except urllib.error.HTTPError as ex:
                time.sleep(5 * 2 ** attempt if ex.code in (429, 500, 503) else 1)
            except Exception:
                time.sleep(2 ** attempt)
        for k, i in enumerate(ids):
            if k in got:
                res[i] = got[k]; cache[ck(items[i])] = list(got[k])
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            json.dump(cache, open(cache_path, "w"))
    return res
