#!/usr/bin/env python3
"""Design-space sweeps for tessera. Prints Markdown tables.

usage: python scripts/sweep.py [--bin build/tessera-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(binary, *args):
    out = subprocess.run([binary, *map(str, args), "--analytical"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"tessera-run {' '.join(map(str, args))} failed:\n{out.stderr}")
    s = json.loads(out.stdout)
    assert s["correct"], s
    return s


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(Path(__file__).resolve().parents[1] / "build" / "tessera-run"))
    binary = ap.parse_args().bin

    print("### Double buffering (GEMM 256x256x256)\n")
    rows = []
    for r, c in [(16, 16), (32, 32), (64, 64)]:
        base = ["--m", 256, "--k", 256, "--n", 256, "--rows", r, "--cols", c, "--sbuf-kb", 512, "--psum-kb", 256]
        s1 = run(binary, *base, "--buffers", 1)
        s2 = run(binary, *base, "--buffers", 2)
        rows.append([f"{r}x{c}", s1["cycles"], s2["cycles"], f"{s1['cycles'] / s2['cycles']:.2f}x",
                     f"{s1['pe_utilization']:.1%}", f"{s2['pe_utilization']:.1%}"])
    print(table(["array", "cycles (1 buf)", "cycles (2 buf)", "speedup", "PE util (1 buf)", "PE util (2 buf)"], rows))

    print("\n### DMA bandwidth sensitivity (32x32 array, GEMM 256x512x256, double buffered)\n")
    rows = []
    for bw in [8, 16, 32, 64, 128, 256]:
        s = run(binary, "--m", 256, "--k", 512, "--n", 256, "--rows", 32, "--cols", 32, "--dma-bw", bw,
                "--sbuf-kb", 512, "--psum-kb", 256)
        e = s["engines"]
        bound = "DMA" if e["dma_in"]["occupancy"] > e["pe"]["occupancy"] else "PE"
        rows.append([bw, s["cycles"], f"{s['pe_utilization']:.1%}", f"{e['dma_in']['occupancy']:.1%}",
                     f"{e['pe']['occupancy']:.1%}", bound])
    print(table(["DMA B/cycle", "cycles", "PE util", "DMA-in busy", "PE busy", "bound by"], rows))

    print("\n### Output tile height (32x32 array, GEMM 512x256x256, double buffered)\n")
    rows = []
    for tm in [8, 16, 32, 64, 128, 256]:
        s = run(binary, "--m", 512, "--k", 256, "--n", 256, "--rows", 32, "--cols", 32, "--tile-m", tm,
                "--sbuf-kb", 512, "--psum-kb", 256)
        rows.append([tm, s["layers"][0]["matmuls"], s["instructions"], s["cycles"], f"{s['pe_utilization']:.1%}"])
    print(table(["tile_m", "matmuls", "instructions", "cycles", "PE util"], rows))


if __name__ == "__main__":
    main()
