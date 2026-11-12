"""Merge a LoRA adapter (a training checkpoint) into its base model for serving.

    python -m statealign.evaluation.merge_lora --base <base dir> --adapter <run>/checkpoint-N --out <dir>

The base model is loaded in bf16 on CPU, the adapter is applied with PeftModel.from_pretrained and merged with
merge_and_unload(); the output is one safetensors file with the tokenizer and generation_config of the base and the
chat template of the adapter. The script checks that every saved LoRA tensor is loaded as saved and that the output
tensors match the base checkpoint's names, shapes and dtypes, and writes merge_info.json (adapter path and sha256, base).
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import torch


def st_header(path):
    """Tensor table of a .safetensors file without loading any data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def st_tensors(model_dir):
    out = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        for k, v in st_header(f).items():
            out[k] = (tuple(v["shape"]), v["dtype"])
    return out


def patch_config(dst: Path):
    """config.json gets a top-level rope_theta if missing; tokenizer_config.json loses extra_special_tokens."""
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if "rope_theta" not in cfg:
        rp = cfg.get("rope_parameters") or {}
        cfg["rope_theta"] = rp.get("rope_theta", 1000000.0)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"  patched config.json: rope_theta = {cfg['rope_theta']}")
    tc_path = dst / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        if "extra_special_tokens" in tc:
            del tc["extra_special_tokens"]
            tc_path.write_text(json.dumps(tc, indent=2))
            print("  patched tokenizer_config.json: removed extra_special_tokens")


def merge_peft(base, adapter, out, max_shard_size="50GB"):
    """PEFT merge_and_unload in bf16 on CPU."""
    import peft
    import transformers
    from peft import PeftModel
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    base, adapter, out = Path(base).resolve(), Path(adapter).resolve(), Path(out).resolve()
    for f in (base / "config.json", adapter / "adapter_config.json", adapter / "adapter_model.safetensors"):
        if not f.exists():
            sys.exit(f"[merge] FAIL missing {f}")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"[merge] FAIL output dir {out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    acfg = json.loads((adapter / "adapter_config.json").read_text())
    print(f"[merge] torch {torch.__version__}  transformers {transformers.__version__}  peft {peft.__version__}")
    print(f"[merge] base={base}")
    print(f"[merge] adapter={adapter}  (r={acfg.get('r')} alpha={acfg.get('lora_alpha')} "
          f"targets={sorted(acfg.get('target_modules') or [])})")
    trained_on = acfg.get("base_model_name_or_path")
    if trained_on and os.path.realpath(trained_on) != str(base):
        print(f"[merge] WARNING adapter_config base_model_name_or_path={trained_on} differs from --base")

    print("[merge] loading base bf16 on CPU ...")
    tf_major, tf_minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (tf_major, tf_minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(str(base), low_cpu_mem_usage=True, **{dtype_kw: torch.bfloat16})
    n_base = model.num_parameters()

    print("[merge] loading adapter ...")
    model = PeftModel.from_pretrained(model, str(adapter))
    # A key-name mismatch would leave lora_B at zero and silently give back the base model.
    file_sd = load_file(str(adapter / "adapter_model.safetensors"))
    model_sd = get_peft_model_state_dict(model)
    missing = sorted(set(file_sd) - set(model_sd))
    unused = sorted(set(model_sd) - set(file_sd))
    differ = [k for k in file_sd if k in model_sd and not torch.equal(file_sd[k].float(), model_sd[k].float().cpu())]
    if missing or unused or differ:
        sys.exit(f"[merge] FAIL adapter not loaded as saved: {len(missing)} keys not in model, {len(unused)} model "
                 f"LoRA keys not in file, {len(differ)} tensors differ; e.g. {(missing + unused + differ)[:3]}")
    b_max = max(float(v.abs().max()) for k, v in file_sd.items() if "lora_B" in k)
    print(f"[merge] adapter check OK: {len(file_sd)} LoRA tensors loaded exactly; max|lora_B| = {b_max:.3e}")
    if b_max == 0.0:
        print("[merge] WARNING all lora_B are zero: the merged model equals the base model")
    del file_sd, model_sd

    print("[merge] merging ...")
    model = model.merge_and_unload()
    left = [n for n, _ in model.named_modules() if "lora" in n.lower()]
    if left:
        sys.exit(f"[merge] FAIL LoRA modules left after merge_and_unload: {left[:3]}")
    n_merged = model.num_parameters()
    bad_dtype = {str(p.dtype) for p in model.parameters() if p.dtype != torch.bfloat16}
    if n_merged != n_base or bad_dtype:
        sys.exit(f"[merge] FAIL merged params {n_merged} vs base {n_base}, non-bf16 dtypes {bad_dtype}")

    print(f"[merge] saving to {out} ...")
    model.save_pretrained(out, safe_serialization=True, max_shard_size=max_shard_size)
    tok = AutoTokenizer.from_pretrained(str(base))
    tok.save_pretrained(out)
    adapter_chat = adapter / "chat_template.jinja"
    if adapter_chat.exists():
        shutil.copy(adapter_chat, out / "chat_template.jinja")
        same = adapter_chat.read_text().strip() == (tok.chat_template or "").strip()
        print(f"  copied chat_template.jinja from adapter ({'identical to' if same else 'DIFFERENT from'} base template)")
    if not (out / "generation_config.json").exists() and (base / "generation_config.json").exists():
        shutil.copy(base / "generation_config.json", out / "generation_config.json")
        print("  copied generation_config.json from base")
    patch_config(out)
    (out / "merge_info.json").write_text(json.dumps({
        "base": str(base), "adapter": str(adapter),
        "adapter_sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "dtype": "bfloat16", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")

    print("[merge] verifying output ...")
    cfg = AutoConfig.from_pretrained(str(out))
    tok2 = AutoTokenizer.from_pretrained(str(out))
    if adapter_chat.exists() and (tok2.chat_template or "").strip() != adapter_chat.read_text().strip():
        sys.exit("[merge] FAIL reloaded tokenizer chat_template differs from the adapter's chat_template.jinja")
    got, ref = st_tensors(out), st_tensors(base)
    n_out = sum(math.prod(s) for s, _ in got.values())
    problems = []
    if set(got) != set(ref):
        problems.append(f"tensor names differ from base: +{sorted(set(got) - set(ref))[:3]} -{sorted(set(ref) - set(got))[:3]}")
    problems += [f"{k}: {got[k]} vs base {ref[k]}" for k in got if k in ref and got[k] != ref[k]][:3]
    if {d for _, d in got.values()} != {"BF16"}:
        problems.append(f"dtypes {sorted({d for _, d in got.values()})}")
    if n_out != n_base:
        problems.append(f"saved params {n_out} != model params {n_base}")
    if problems:
        sys.exit("[merge] FAIL " + "; ".join(problems))
    gen = json.loads((out / "generation_config.json").read_text())
    print(f"  config: {cfg.model_type} {cfg.architectures} layers={cfg.num_hidden_layers} "
          f"rope_theta={getattr(cfg, 'rope_theta', None)} dtype={getattr(cfg, 'dtype', None) or getattr(cfg, 'torch_dtype', None)}")
    print(f"  generation_config: temperature={gen.get('temperature')} top_p={gen.get('top_p')} "
          f"top_k={gen.get('top_k')} eos={gen.get('eos_token_id')}")
    print(f"  tokenizer: {type(tok2).__name__} vocab={len(tok2)} eos={tok2.eos_token} pad={tok2.pad_token}")
    print(f"  parameters: {n_out:,} in {len(got)} tensors (all BF16, names/shapes match base)")

    print("\n[merge] output files:")
    total = 0
    for f in sorted(out.iterdir()):
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name:<40s} {sz / 1e9:6.2f} GB" if sz > 1e9 else f"  {f.name:<40s} {sz:>12,} bytes")
    print(f"  total: {total / 1e9:.2f} GB")
    print(f"MERGE_DONE {out} params={n_out} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shard-size", default="50GB", help="the default gives a single model.safetensors")
    a = ap.parse_args()
    merge_peft(a.base, a.adapter, a.out, a.max_shard_size)


if __name__ == "__main__":
    main()
