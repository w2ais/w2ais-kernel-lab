# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Plot the extended capture as compact SVGs for the shared Rough.js renderer."""
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
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "gb10-extended",
                     "font.family": "DejaVu Sans", "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#202020", "text.color": "#202020",
                     "xtick.color": "#626262", "ytick.color": "#626262",
                     "axes.edgecolor": "#626262", "axes.linewidth": .8})
GREEN, BLUE, INK = "#2e7253", "#386a94", "#202020"


def save(fig, name, title, description):
    path = args.output / name
    fig.savefig(path, format="svg", metadata={"Date": None, "Title": title, "Description": description})
    plt.close(fig)
    svg = re.sub(r"font-family:[^;\"]*;?", "", path.read_text())
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.fromstring(svg)
    for element in tree.iter():
        remainder = []
        for rule in element.attrib.pop("style", "").split(";"):
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


fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
for ax, op, title, color in zip(axes, ["ffma", "exp2"], ["MMA + FP32 FMA", "MMA + exp2"], [GREEN, BLUE]):
    rows = [p for p in data["mixed"] if p["op"] == op]
    for key, label, style, shade in [("serial_sum", "Separate times added", ":", "#777777"),
                                    ("mixed", "Interleaved loop", "-", color),
                                    ("ideal_overlap", "Longer time alone", "--", INK)]:
        ax.plot([1,4,8], [p[key] for p in rows], label=label, linestyle=style, color=shade,
                linewidth=1.6, marker="o", markersize=4)
    ax.set_xticks([1,4,8], labels=["1","4","8"])
    ax.set_xlabel("Scalar instructions per MMA", labelpad=9, fontsize=12)
    ax.set_title(title, loc="left", fontsize=16, pad=13)
axes[0].set_ylim(0, max(p["serial_sum"] for p in data["mixed"])*1.08)
axes[0].set_yticks([0,4,8,12,16,20])
axes[0].set_ylabel("SM cycles per base MMA", labelpad=10, fontsize=12)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
           bbox_to_anchor=(.53,0))
fig.subplots_adjust(left=.1, right=.98, top=.85, bottom=.3, wspace=.15)
save(fig, "mixed_overlap.svg", "Independent arithmetic overlaps, but is not free",
     "One CTA with 16 warps, four MMA fragments per warp. The colored line measures an interleaved loop; the dotted line adds separate MMA-only and scalar-only measurements. Five-round average of medians.")

fig, ax = plt.subplots(figsize=(9.5,3.8))
sizes = [4096,65536,1048576,8388608,67108864,268435456]
for op, label, color in [("ca",".ca — cache at all levels",GREEN), ("cg",".cg — bypass L1",BLUE)]:
    rows = [next(p for p in data["points"] if p["family"]=="memory" and p["op"]==op and p["size"]==s) for s in sizes]
    ax.plot(sizes, [p["cycles_per_work"] for p in rows], color=color, marker="o", linewidth=1.6, label=label)
ax.set_xscale("log", base=2)
ax.set_xticks(sizes, labels=["4 KiB","64 KiB","1 MiB","8 MiB","64 MiB","256 MiB"])
ax.set_xlabel("Address span · one pointer every 128 bytes", labelpad=10, fontsize=13)
ax.set_ylabel("Cycles / dependent load", labelpad=10, fontsize=13)
ax.set_ylim(bottom=0)
ax.legend(frameon=False, loc="upper left", fontsize=12)
fig.subplots_adjust(left=.13, right=.97, top=.95, bottom=.23)
save(fig, "memory_latency.svg", "A dependent pointer follows the memory hierarchy",
     "Randomized pointer cycles, one active lane, two full warmups. The entire address span is visited in every sample. Address span is not cache footprint: each node occupies one sector of a 128-byte interval. Timings include dependency and loop scheduling.")

fig, ax = plt.subplots(figsize=(9.5,3.5))
rows = data["shared"]
ax.bar(range(7), [r["cycles_per_step"] for r in rows], width=.57,
       color=[GREEN, GREEN, BLUE, BLUE, BLUE, BLUE, BLUE], alpha=.8)
for i, r in enumerate(rows):
    ax.text(i, r["cycles_per_step"]+2, f'{r["cycles_per_step"]:.1f}', ha="center", fontsize=12)
ax.set_xticks(range(7), labels=["Broadcast","1","2","4","8","16","32"])
ax.set_ylabel("Cycles / dependent LDS", labelpad=10, fontsize=13)
ax.set_xlabel("Stride between lanes, in 32-bit words", labelpad=10, fontsize=13)
ax.set_ylim(0,max(r["cycles_per_step"] for r in rows)*1.19)
fig.subplots_adjust(left=.12, right=.98, top=.95, bottom=.25)
save(fig, "shared_banks.svg", "Bank conflicts lengthen the dependent load chain",
     "One warp, three-length recurrence fits, five rounds. Stride zero broadcasts one word; stride one uses distinct banks. Stride 32 addresses different words of the same bank.")

fig, ax = plt.subplots(figsize=(9.5,4))
for op,label,color,style in [("vector","Vector LDG + STS", "#777777", ":"),
                            ("cp_async","cp.async",GREEN,"-"),
                            ("bulk","Bulk load", "#a5713a","--"),
                            ("tma2d","2D TMA load",BLUE,"-"),
                            ("store","Bulk store",INK,"--")]:
    rows=[next(p for p in data["points"] if p["family"]=="copy" and p["op"]==op
               and p["size"]==4096 and p["depth"]==d) for d in [1,2,4,8]]
    ax.plot([1,2,4,8], [p["work_per_cycle"] for p in rows], color=color, linestyle=style,
            marker="o", linewidth=1.6, label=label)
ax.set_xscale("log", base=2)
ax.set_xticks([1,2,4,8], labels=["1","2","4","8"])
ax.set_xlabel("Transfers per batch · 4 KiB each", labelpad=10, fontsize=13)
ax.set_ylabel("Requested bytes / SM cycle", labelpad=10, fontsize=13)
ax.set_ylim(bottom=0)
fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=3, loc="lower center",
           bbox_to_anchor=(.54,-.01), fontsize=11)
fig.subplots_adjust(left=.13,right=.98,top=.95,bottom=.35)
save(fig, "async_batches.svg", "Batching amortizes asynchronous copy overhead",
     "One four-warp CTA, warm cache, 128 fully drained batches. cp.async uses commit/wait groups; bulk and tensor loads use mbarriers; bulk stores use full wait_group, not read-only completion. This is not whole-GPU bandwidth or a GEMM pipeline.")
