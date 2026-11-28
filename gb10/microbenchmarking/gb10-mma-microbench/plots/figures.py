# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Render standalone data figures; Astro applies the shared Rough.js renderer."""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("capture", type=Path)
parser.add_argument("--output", type=Path, help="Figure directory (default: CAPTURE/figures)")
args = parser.parse_args()
args.output = args.output or args.capture / "figures"
args.output.mkdir(parents=True, exist_ok=True)
data = json.loads((args.capture / "analysis.json").read_text())
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "gb10-mma",
                     "font.family": "DejaVu Sans", "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#202020", "text.color": "#202020",
                     "xtick.color": "#626262", "ytick.color": "#626262",
                     "axes.edgecolor": "#626262", "axes.linewidth": .8})


def save(fig, name, title, description):
    path = args.output / name
    fig.savefig(path, format="svg", metadata={"Date": None, "Title": title,
                                             "Description": description})
    plt.close(fig)
    svg = path.read_text()
    # Keep text as text. The shared renderer supplies the local handwritten font.
    svg = re.sub(r"font-family:[^;\"]*;?", "", svg)
    # Matplotlib writes paint into inline styles. The shared Rough.js renderer
    # reads SVG presentation attributes, including inherited group attributes.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.fromstring(svg)
    for element in tree.iter():
        styles = element.attrib.pop("style", "")
        remainder = []
        for rule in styles.split(";"):
            if ":" not in rule:
                continue
            key, value = map(str.strip, rule.split(":", 1))
            if key in {"fill", "stroke", "stroke-width", "fill-opacity", "stroke-opacity",
                       "stroke-dasharray", "font-size", "text-anchor", "opacity"}:
                element.set(key, value)
            else:
                remainder.append(rule)
        if remainder:
            element.set("style", ";".join(remainder))
    path.write_text(ET.tostring(tree, encoding="unicode"))


fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
for ax, op, label, color in zip(axes, ["bf16", "nvfp4"], ["BF16 · K = 16", "NVFP4 · K = 64"], ["#2e7253", "#386a94"]):
    for chains, style, shade in [(1, ":", "#777777"), (4, "-", color), (8, "--", color)]:
        points = [next(p for p in data["points"] if
                       (p["op"],p["warps"],p["chains"],p["unroll"],p["iterations"],p["variant"])
                       == (op,str(w),str(chains),"16","2048","0")) for w in [1,2,4,8,16,32]]
        ax.plot([int(p["warps"]) for p in points], [p["mma_per_sm_cycle"] for p in points],
                linestyle=style, marker="o", markersize=3.5, color=shade, linewidth=1.6,
                label=f"{chains} independent {'fragment' if chains == 1 else 'fragments'} / warp")
    ax.axhline(.25, color="#202020", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1,2,4,8,16,32], labels=[1,2,4,8,16,32])
    ax.set_yticks([0,.05,.10,.15,.20,.25], labels=["0",".05",".10",".15",".20",".25"])
    ax.set_ylim(0,.268)
    ax.set_xlabel("Warps in one CTA", labelpad=9)
    ax.set_title(label, loc="left", fontsize=16, pad=13)
axes[0].set_ylabel("MMA / SM / cycle", labelpad=12)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=12,
           bbox_to_anchor=(.55,-.015))
fig.subplots_adjust(left=.11, right=.98, top=.85, bottom=.34, wspace=.15)
save(fig, "mma_saturation.svg", "Independent work fills the Tensor Core pipeline",
     "Measured MMA per SM cycle versus active warps. Curves use one, four, or eight independent accumulator fragments per warp. Black line: 0.25 reference rate. Five rounds; one CTA.")

labels = {"fp16_k8":"FP16 · K=8", "fp16":"FP16 · K=16", "bf16":"BF16 · K=16",
          "tf32":"TF32 · K=8", "fp4_k32":"E2M1 padded · K=32", "mxfp4":"MXFP4 packed · K=64", "nvfp4":"NVFP4 packed · K=64"}
fig, ax = plt.subplots(figsize=(9.5,4.4))
for y, row in enumerate(data["summary"]):
    value = row["flops_per_sm_cycle"]
    color = "#386a94" if "fp4" in row["op"] else "#2e7253"
    ax.barh(y, value, height=.57, color=color, alpha=.8)
    ax.text(value+65, y, f"{value:,.1f}", va="center", fontsize=11)
ax.set_yticks(range(7), labels=[labels[r["op"]] for r in data["summary"]])
ax.invert_yaxis()
ax.set_xlim(0,4800)
ax.set_xticks([0,1024,2048,3072,4096],labels=["0","1,024","2,048","3,072","4,096"])
ax.set_xlabel("Measured FLOP / SM / cycle", labelpad=10)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=10)
fig.subplots_adjust(left=.3,right=.97,bottom=.19,top=.96)
save(fig, "mma_flops.svg", "Similar instruction rates, different arithmetic throughput",
     "Measured dense FLOPs per SM cycle, computed from 2MNK times measured MMA rate. Reporting configuration: 16 warps, four independent fragments per warp. FP32 accumulation.")
