#!/usr/bin/env python3
"""Greedy reference continuations from llama.cpp CUDA (llama-server with
token-array prompts, so both runners see identical prompt ids).
Usage: llama_ref.py <llama-server> <model.gguf> <n_predict> <out.json> <prompt.csv>...
Env: LLAMA_REF_PORT (default 18093), LLAMA_REF_EXTRA (extra server args)."""
import json, os, subprocess, sys, time, urllib.request

server, model, n_predict, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompts = sys.argv[5:]
port = int(os.environ.get("LLAMA_REF_PORT", "18093"))
extra = os.environ.get("LLAMA_REF_EXTRA", "").split()
cmd = [server, "-m", model, "-ngl", "99", "--port", str(port), "--host", "127.0.0.1",
       "-c", "2048", "-np", "1", "--no-warmup"] + extra
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(out + ".server.log", "w"))
try:
    for _ in range(240):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if json.load(r).get("status") == "ok":
                    break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        raise SystemExit("llama-server did not become healthy")
    results = {}
    for path in prompts:
        ids = [int(x) for x in open(path).read().strip().split(",")]
        body = {"prompt": ids, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
                "cache_prompt": False, "return_tokens": True, "n_probs": 5,
                "samplers": ["top_k"], "repeat_penalty": 1.0, "ignore_eos": True}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.load(r)
        toks = resp.get("tokens")
        if toks is None:
            toks = [p["id"] for p in resp.get("completion_probabilities", [])]
        results[os.path.basename(path)] = {"prompt_len": len(ids), "tokens": toks,
                                           "content": resp.get("content"),
                                           "timings": resp.get("timings")}
        print(os.path.basename(path), len(ids), toks)
    json.dump(results, open(out, "w"), indent=1)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
