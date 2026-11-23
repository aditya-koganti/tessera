// SPDX-License-Identifier: MIT
#include "tessera/compiler.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

namespace tessera {

namespace {

int ceil_div(int a, int b) { return (a + b - 1) / b; }

std::string tag(std::size_t layer, const char* what, int m0, int k0, int n0) {
  return "L" + std::to_string(layer) + " " + what + " m" + std::to_string(m0) + " k" + std::to_string(k0) +
         " n" + std::to_string(n0);
}

// Wait for `sem >= value`, skipping conditions that are trivially true.
void wait_on(std::vector<Wait>& waits, int sem, std::int64_t value) {
  if (value > 0) waits.push_back(Wait{sem, value});
}

}  // namespace

Program compile(const HwConfig& cfg, const std::vector<GemmLayer>& layers, const CompileOptions& opts,
                std::vector<LayerPlan>* plans) {
  cfg.validate();
  if (layers.empty()) throw CompileError("no layers to compile");
  if (opts.buffers < 1 || opts.buffers > 4) throw CompileError("buffers must be in [1, 4]");
  if (opts.tile_m < 0) throw CompileError("tile_m must be >= 0");

  const int R = cfg.array_rows;
  const int C = cfg.array_cols;
  const int B = opts.buffers;

  Program prog;
  const int s_a = prog.add_semaphore("a_loaded");
  const int s_w = prog.add_semaphore("w_loaded");
  const int s_mm = prog.add_semaphore("mm_done");
  const int s_act = prog.add_semaphore("act_done");
  const int s_st = prog.add_semaphore("store_done");

  std::int64_t it = 0;  // global matmul counter (indexes a_loaded / w_loaded / mm_done)
  std::int64_t ot = 0;  // global output-tile counter (indexes act_done / store_done)
  if (plans != nullptr) plans->clear();

  for (std::size_t li = 0; li < layers.size(); ++li) {
    const GemmLayer& L = layers[li];
    if (L.m < 1 || L.k < 1 || L.n < 1) throw CompileError("layer " + std::to_string(li) + ": dimensions must be >= 1");
    if (L.shift < 0 || L.shift > 31) throw CompileError("layer " + std::to_string(li) + ": shift must be in [0, 31]");
    const int esize = L.out_int8 ? 1 : 4;

    // On-chip budget for one tile row: B A-slots of R bytes, B output slots of C*esize bytes,
    // and B PSUM slots of C int32s. Weight slots are a fixed B * R * C bytes.
    const std::size_t w_bytes = static_cast<std::size_t>(B) * R * C;
    const std::size_t sbuf_per_row = static_cast<std::size_t>(B) * (R + static_cast<std::size_t>(C) * esize);
    const std::size_t psum_per_row = static_cast<std::size_t>(B) * C * sizeof(std::int32_t);
    const std::size_t max_sbuf_rows = cfg.sbuf_bytes > w_bytes ? (cfg.sbuf_bytes - w_bytes) / sbuf_per_row : 0;
    const std::size_t max_psum_rows = cfg.psum_bytes / psum_per_row;
    const std::size_t fit = std::min(max_sbuf_rows, max_psum_rows);

    const std::size_t want = opts.tile_m > 0 ? static_cast<std::size_t>(opts.tile_m) : std::size_t{512};
    const int tm = static_cast<int>(std::min({want, static_cast<std::size_t>(L.m), fit}));
    if (tm < 1 || (opts.tile_m > 0 && static_cast<std::size_t>(std::min(opts.tile_m, L.m)) > fit)) {
      throw CompileError("layer " + std::to_string(li) + ": tile does not fit on chip (max tile_m=" +
                         std::to_string(fit) + ", SBUF " + std::to_string(cfg.sbuf_bytes) + " B, PSUM " +
                         std::to_string(cfg.psum_bytes) + " B)");
    }

    // SBUF layout: [A slots][W slots][output slots]; PSUM: [accumulator slots].
    const std::size_t a_slot = static_cast<std::size_t>(tm) * R;
    const std::size_t w_slot = static_cast<std::size_t>(R) * C;
    const std::size_t o_slot = static_cast<std::size_t>(tm) * C * esize;
    const std::size_t p_slot = static_cast<std::size_t>(tm) * C;
    const std::size_t a_base = 0;
    const std::size_t w_base = a_base + B * a_slot;
    const std::size_t o_base = w_base + B * w_slot;

    const int tiles_m = ceil_div(L.m, tm);
    const int tiles_k = ceil_div(L.k, R);
    const int tiles_n = ceil_div(L.n, C);
    const std::int64_t layer_start_outputs = ot;
    bool first_load = true;

    for (int nt = 0; nt < tiles_n; ++nt) {
      for (int mt = 0; mt < tiles_m; ++mt) {
        const std::int64_t j = ot;
        const int m0 = mt * tm;
        const int ms = std::min(tm, L.m - m0);
        const int n0 = nt * C;
        const int ns = std::min(C, L.n - n0);
        const std::size_t ps = static_cast<std::size_t>(j % B) * p_slot;
        const std::size_t os = o_base + static_cast<std::size_t>(j % B) * o_slot;

        for (int kt = 0; kt < tiles_k; ++kt) {
          const std::int64_t i = it;
          const int k0 = kt * R;
          const int ks = std::min(R, L.k - k0);
          const std::size_t as = a_base + static_cast<std::size_t>(i % B) * a_slot;
          const std::size_t ws = w_base + static_cast<std::size_t>(i % B) * w_slot;

          // Loads into slot i%B may start once matmul i-B (the previous user) has finished.
          std::vector<Wait> load_waits;
          wait_on(load_waits, s_mm, i - B + 1);
          if (first_load && li > 0) {
            // Layer inputs are the previous layer's outputs in DRAM.
            wait_on(load_waits, s_st, layer_start_outputs);
          }
          first_load = false;

          Instruction la;
          la.engine = Engine::DmaIn;
          la.op = DmaOp{Space::Dram, Space::Sbuf, L.a_dram + static_cast<std::size_t>(m0) * L.k + k0, as, ms, ks, L.k, R};
          la.waits = load_waits;
          la.signal = s_a;
          la.label = tag(li, "load A", m0, k0, n0);
          prog.instrs.push_back(std::move(la));

          Instruction lw;
          lw.engine = Engine::DmaIn;
          lw.op = DmaOp{Space::Dram, Space::Sbuf, L.w_dram + static_cast<std::size_t>(k0) * L.n + n0, ws, ks, ns, L.n, C};
          wait_on(lw.waits, s_mm, i - B + 1);
          lw.signal = s_w;
          lw.label = tag(li, "load W", m0, k0, n0);
          prog.instrs.push_back(std::move(lw));

          Instruction mm;
          mm.engine = Engine::Pe;
          mm.op = MatMulOp{as, R, ws, C, ps, C, ms, ks, ns, kt > 0};
          wait_on(mm.waits, s_a, i + 1);
          wait_on(mm.waits, s_w, i + 1);
          if (kt == 0) wait_on(mm.waits, s_act, j - B + 1);  // PSUM slot drained by ACT
          mm.signal = s_mm;
          mm.label = tag(li, "matmul", m0, k0, n0);
          prog.instrs.push_back(std::move(mm));
          ++it;
        }

        Instruction act;
        act.engine = Engine::Act;
        act.op = ActOp{ps, C, os, C * esize, ms, ns, L.relu, L.shift, L.out_int8};
        wait_on(act.waits, s_mm, it);              // all K partial sums accumulated
        wait_on(act.waits, s_st, j - B + 1);       // output slot written back to DRAM
        act.signal = s_act;
        act.label = tag(li, "epilogue", m0, 0, n0);
        prog.instrs.push_back(std::move(act));

        Instruction st;
        st.engine = Engine::DmaOut;
        st.op = DmaOp{Space::Sbuf, Space::Dram, os,
                      L.c_dram + (static_cast<std::size_t>(m0) * L.n + n0) * esize, ms, ns * esize, C * esize,
                      L.n * esize};
        wait_on(st.waits, s_act, j + 1);
        st.signal = s_st;
        st.label = tag(li, "store C", m0, 0, n0);
        prog.instrs.push_back(std::move(st));
        ++ot;
      }
    }

    if (plans != nullptr) {
      LayerPlan plan;
      plan.tile_m = tm;
      plan.tiles_m = tiles_m;
      plan.tiles_k = tiles_k;
      plan.tiles_n = tiles_n;
      plan.matmuls = tiles_m * tiles_k * tiles_n;
      plan.sbuf_bytes_used = o_base + B * o_slot;
      plan.psum_bytes_used = B * p_slot * sizeof(std::int32_t);
      plans->push_back(plan);
    }
  }
  return prog;
}

}  // namespace tessera
