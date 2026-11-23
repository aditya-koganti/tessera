"""End-to-end tests that drive tessera-run and check results independently of the C++ golden model."""

import json

import pytest


def matmul(a, w):
    k = len(w)
    n = len(w[0])
    return [[sum(row[p] * w[p][j] for p in range(k)) for j in range(n)] for row in a]


def requantize(x, shift):
    if shift > 0:
        x = (x + (1 << (shift - 1))) >> shift  # Python >> is floor division by 2**shift
    return max(-128, min(127, x))


def python_reference(dump):
    x = dump["a"]
    for layer in dump["layers"]:
        acc = matmul(x, layer["w"])
        if layer["relu"]:
            acc = [[max(v, 0) for v in row] for row in acc]
        if layer["out_int8"]:
            acc = [[requantize(v, layer["shift"]) for v in row] for row in acc]
        x = acc
    return x


@pytest.mark.parametrize(
    "args",
    [
        ["--m", 13, "--k", 21, "--n", 9],
        ["--m", 16, "--k", 16, "--n", 16, "--relu"],
        ["--m", 10, "--k", 40, "--n", 33, "--relu", "--shift", 10, "--int8-out"],
        ["--m", 7, "--k", 5, "--n", 3, "--buffers", 1, "--rows", 4, "--cols", 4],
        ["--m", 12, "--mlp", "24,20,6", "--rows", 8, "--cols", 8],
    ],
)
def test_output_matches_independent_python_reference(tessera, tmp_path, args):
    dump_path = tmp_path / "dump.json"
    stats = tessera(*args, "--dump", dump_path)
    assert stats["correct"] is True
    dump = json.loads(dump_path.read_text())
    assert dump["output"] == python_reference(dump)


def test_double_buffering_hides_dma_latency(tessera):
    shape = ["--m", 128, "--k", 256, "--n", 128, "--analytical"]
    serial = tessera(*shape, "--buffers", 1)
    double = tessera(*shape, "--buffers", 2)
    assert serial["correct"] and double["correct"]
    assert double["cycles"] < 0.75 * serial["cycles"]
    assert double["pe_utilization"] > serial["pe_utilization"]


def test_cycle_accurate_pe_matches_analytical_model(tessera):
    shape = ["--m", 40, "--k", 48, "--n", 40, "--rows", 8, "--cols", 8]
    exact = tessera(*shape)
    fast = tessera(*shape, "--analytical")
    assert exact["cycles"] == fast["cycles"]
    assert exact["correct"] and fast["correct"]


@pytest.mark.parametrize("rows,cols", [(4, 4), (8, 16), (16, 16), (32, 32)])
def test_utilization_and_engine_accounting(tessera, rows, cols):
    s = tessera("--m", 64, "--k", 96, "--n", 80, "--rows", rows, "--cols", cols, "--analytical")
    assert s["correct"]
    assert 0.0 < s["pe_utilization"] <= 1.0
    assert s["macs"] == 64 * 96 * 80
    for engine in s["engines"].values():
        assert 0 <= engine["busy_cycles"] <= s["cycles"]
    # Every element of C is written exactly once (int32 output).
    assert s["dma_bytes"]["out"] == 64 * 80 * 4


def test_trace_is_valid_and_engines_never_overlap_themselves(tessera, tmp_path):
    trace_path = tmp_path / "trace.json"
    stats = tessera("--m", 48, "--k", 64, "--n", 48, "--rows", 8, "--cols", 8, "--trace", trace_path)
    events = [e for e in json.loads(trace_path.read_text())["traceEvents"] if e["ph"] == "X"]
    assert len(events) == stats["instructions"]
    by_engine = {}
    for e in events:
        by_engine.setdefault(e["tid"], []).append((e["ts"], e["ts"] + e["dur"]))
    for spans in by_engine.values():
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert start >= end
    assert max(end for spans in by_engine.values() for _, end in spans) == stats["cycles"]


def test_tile_that_does_not_fit_is_a_clean_error(tessera):
    proc = tessera("--m", 64, "--k", 16, "--n", 16, "--tile-m", 64, "--psum-kb", 1, check=False)
    assert proc.returncode == 1
    assert "does not fit" in proc.stderr
