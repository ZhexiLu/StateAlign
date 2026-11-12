"""A serving view of a model directory: a new directory with a patched config.json and symlinks to every other file
(the model's own config is never edited). BFCL multi-turn is run with a 65,536-token context; for Qwen3 models this
is YaRN RoPE scaling with factor 2 (rope_type yarn, original_max_position_embeddings 32768):

    python -m statealign.evaluation.serving_view --model <dir> --out <view dir> --yarn 2
"""
import argparse
import json
import os
import shutil


def make_view(model, out, yarn=None):
    model = os.path.realpath(model)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for f in os.listdir(model):
        if f != "config.json":
            os.symlink(os.path.realpath(os.path.join(model, f)), os.path.join(out, f))
    c = json.load(open(os.path.join(model, "config.json")))
    if yarn:
        c["rope_scaling"] = {"rope_type": "yarn", "factor": float(yarn), "original_max_position_embeddings": 32768}
    json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yarn", type=float, default=None, help="YaRN factor (2 = 65,536-token context for Qwen3)")
    a = ap.parse_args()
    print(make_view(a.model, a.out, a.yarn))


if __name__ == "__main__":
    main()
