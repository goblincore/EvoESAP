# goblincore/EvoESAP fork

This branch (`goblincore-fixes`) layers a set of build fixes and dependency
simplifications on top of `ZongfangLiu/EvoESAP`. Tested 2026-05-12 on a fresh
RunPod RTX 4090 with `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

## What's changed vs upstream

1. **`transformers`**: pinned to git source (`huggingface/transformers@main`)
   instead of `==5.0.0dev` which is not on PyPI.
2. **`torch`**: bumped `2.7.1` → `2.9.1` to match the vllm fork's requirement.
3. **`vllm`**: switched from PyPI `==0.13.0` (caps `transformers<5`, conflicts
   with our git source) to `git+https://github.com/goblincore/vllm.git`
   (goblincore-fixes branch — same as ZongfangLiu/vllm but with cap removed
   and static version set so setuptools_scm doesn't crash on tagless clones).
4. **Eval suite stripped**: removed `evalplus[vllm]`, `lm-eval[vllm,api]`,
   `livecodebench`, `crfm-helm`, `evalscope` deps. Search + materialise paths
   don't touch them. Run agentic acceptance bench externally if needed.
5. **`.gitmodules` removed**: upstream's was orphaned (no gitlinks committed).
   With the eval suite stripped, we no longer need any `third-party/` deps.

## Install

```bash
git clone https://github.com/goblincore/EvoESAP.git
cd EvoESAP
git checkout goblincore-fixes
curl -LsSf https://astral.sh/uv/install.sh | sh
bash scripts/build.sh
source .venv/bin/activate
```

Expect ~15–30 min for the install (downloads torch ~3GB, builds flash-attn
from source, etc.). No manual patching, no `third-party/` clones, no env vars.

## Upstreaming

The fixes in (1), (2), (3) are general-purpose bug fixes worth PRing to
upstream `ZongfangLiu/EvoESAP` and `ZongfangLiu/vllm`. Fix (4) is opinionated
(some users will want the eval suite). Fix (5) follows from (4).

See `Claude Notes/Research/2026-05-11-ream-v2-evoesap-coding-runbook.md`
§"EvoESAP build gotchas" for the full investigation that led to these.
