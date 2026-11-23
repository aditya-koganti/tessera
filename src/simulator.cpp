// SPDX-License-Identifier: MIT
#include "tessera/simulator.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <ostream>
#include <sstream>
#include <type_traits>

#include "tessera/reference.hpp"

namespace tessera {

const char* engine_name(Engine e) {
  switch (e) {
    case Engine::DmaIn: return "dma_in";
    case Engine::DmaOut: return "dma_out";
    case Engine::Pe: return "pe";
    case Engine::Act: return "act";
  }
  return "unknown";
}

void HwConfig::validate() const {
  auto fail = [](const char* what) { throw std::invalid_argument(std::string("HwConfig: ") + what); };
  if (array_rows < 1 || array_cols < 1) fail("array_rows and array_cols must be >= 1");
  if (array_rows > 1024 || array_cols > 1024) fail("array dimensions must be <= 1024");
  if (sbuf_bytes == 0 || psum_bytes < sizeof(std::int32_t) || dram_bytes == 0) fail("memories must be non-empty");
  if (dma_bytes_per_cycle < 1) fail("dma_bytes_per_cycle must be >= 1");
  if (dma_latency_cycles < 0 || act_latency_cycles < 0) fail("latencies must be >= 0");
  if (act_lanes < 1) fail("act_lanes must be >= 1");
}

double SimStats::occupancy(Engine e) const {
  if (total_cycles == 0) return 0.0;
  return static_cast<double>(engines[static_cast<int>(e)].busy_cycles) / static_cast<double>(total_cycles);
}

double SimStats::pe_utilization(const HwConfig& cfg) const {
  if (total_cycles == 0) return 0.0;
  const double peak = static_cast<double>(cfg.array_rows) * cfg.array_cols * static_cast<double>(total_cycles);
  return static_cast<double>(useful_macs) / peak;
}

namespace {

enum class Region : int { Dram = 0, Sbuf = 1, Psum = 2 };

struct Access {
  Region region;
  std::size_t lo;
  std::size_t hi;  // exclusive
  bool write;
};

Region region_of(Space s) { return s == Space::Dram ? Region::Dram : Region::Sbuf; }

bool conflicts(const Access& a, const Access& b) {
  return a.region == b.region && a.lo < b.hi && b.lo < a.hi && (a.write || b.write);
}

// Conservative footprint: the bounding interval of each strided region.
std::vector<Access> footprint(const Instruction& in) {
  std::vector<Access> acc;
  auto add = [&](Region r, std::size_t lo, std::size_t len, bool write) {
    if (len > 0) acc.push_back({r, lo, lo + len, write});
  };
  std::visit(
      [&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, DmaOp>) {
          add(region_of(op.src_space), op.src, region_span(op.rows, op.row_bytes, op.src_stride), false);
          add(region_of(op.dst_space), op.dst, region_span(op.rows, op.row_bytes, op.dst_stride), true);
        } else if constexpr (std::is_same_v<T, MatMulOp>) {
          add(Region::Sbuf, op.act_addr, region_span(op.m, op.k, op.act_stride), false);
          add(Region::Sbuf, op.w_addr, region_span(op.k, op.n, op.w_stride), false);
          add(Region::Psum, op.psum_addr, region_span(op.m, op.n, op.psum_stride), true);
        } else {
          const int esize = op.out_int8 ? 1 : 4;
          add(Region::Psum, op.psum_addr, region_span(op.rows, op.cols, op.psum_stride), false);
          add(Region::Sbuf, op.dst_addr, region_span(op.rows, op.cols * esize, op.dst_stride), true);
        }
      },
      in.op);
  return acc;
}

bool engine_accepts(Engine e, const Op& op) {
  switch (e) {
    case Engine::DmaIn:
    case Engine::DmaOut: return std::holds_alternative<DmaOp>(op);
    case Engine::Pe: return std::holds_alternative<MatMulOp>(op);
    case Engine::Act: return std::holds_alternative<ActOp>(op);
  }
  return false;
}

std::string describe(const Program& p, int idx) {
  const Instruction& in = p.instrs[static_cast<std::size_t>(idx)];
  return "#" + std::to_string(idx) + " " + engine_name(in.engine) + " '" + in.label + "'";
}

std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return (a + b - 1) / b; }

void json_escape(std::ostream& os, const std::string& s) {
  for (char c : s) {
    if (c == '"' || c == '\\') os << '\\' << c;
    else if (static_cast<unsigned char>(c) < 0x20) os << ' ';
    else os << c;
  }
}

}  // namespace

Simulator::Simulator(HwConfig cfg)
    : cfg_((cfg.validate(), cfg)),
      dram_("DRAM", cfg.dram_bytes),
      sbuf_("SBUF", cfg.sbuf_bytes),
      psum_("PSUM", cfg.psum_bytes / sizeof(std::int32_t)),
      array_(cfg.array_rows, cfg.array_cols) {}

std::int64_t Simulator::execute(const Instruction& in, SimStats& stats) {
  return std::visit(
      [&](const auto& op) -> std::int64_t {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, DmaOp>) {
          if (op.rows < 0 || op.row_bytes < 0 || op.src_stride < 0 || op.dst_stride < 0) {
            throw std::invalid_argument("DMA with negative shape: " + in.label);
          }
          ByteMemory& src = op.src_space == Space::Dram ? dram_ : sbuf_;
          ByteMemory& dst = op.dst_space == Space::Dram ? dram_ : sbuf_;
          src.check(op.src, region_span(op.rows, op.row_bytes, op.src_stride));
          dst.check(op.dst, region_span(op.rows, op.row_bytes, op.dst_stride));
          for (int r = 0; r < op.rows; ++r) {
            const std::size_t so = op.src + static_cast<std::size_t>(r) * op.src_stride;
            const std::size_t d = op.dst + static_cast<std::size_t>(r) * op.dst_stride;
            std::memmove(dst.at(d, op.row_bytes), src.at(so, op.row_bytes), static_cast<std::size_t>(op.row_bytes));
          }
          const std::int64_t bytes = static_cast<std::int64_t>(op.rows) * op.row_bytes;
          if (op.dst_space == Space::Dram) stats.dma_bytes_out += bytes;
          else stats.dma_bytes_in += bytes;
          return cfg_.dma_latency_cycles + ceil_div(bytes, cfg_.dma_bytes_per_cycle);
        } else if constexpr (std::is_same_v<T, MatMulOp>) {
          auto* a = reinterpret_cast<const std::int8_t*>(
              sbuf_.at(op.act_addr, region_span(op.m, op.k, op.act_stride)));
          auto* w = reinterpret_cast<const std::int8_t*>(
              sbuf_.at(op.w_addr, region_span(op.k, op.n, op.w_stride)));
          std::int32_t* out = psum_.at(op.psum_addr, region_span(op.m, op.n, op.psum_stride));
          MatMulTiming t;
          if (cfg_.cycle_accurate_pe) {
            t = array_.run(a, op.act_stride, w, op.w_stride, op.m, op.k, op.n, out, op.psum_stride,
                           op.accumulate);
          } else {
            t = SystolicArray::analytical(cfg_.array_rows, cfg_.array_cols, op.m, op.k, op.n);
            for (int i = 0; i < op.m; ++i) {
              for (int j = 0; j < op.n; ++j) {
                std::int32_t acc = 0;
                for (int p = 0; p < op.k; ++p) {
                  acc += static_cast<std::int32_t>(a[i * op.act_stride + p]) * w[p * op.w_stride + j];
                }
                std::int32_t& dst = out[i * op.psum_stride + j];
                dst = op.accumulate ? dst + acc : acc;
              }
            }
          }
          stats.useful_macs += t.useful_macs;
          return t.total();
        } else {
          if (op.shift < 0 || op.shift > 31) throw std::invalid_argument("ACT shift must be in [0, 31]");
          const int esize = op.out_int8 ? 1 : 4;
          const std::int32_t* src = psum_.at(op.psum_addr, region_span(op.rows, op.cols, op.psum_stride));
          sbuf_.check(op.dst_addr, region_span(op.rows, op.cols * esize, op.dst_stride));
          for (int r = 0; r < op.rows; ++r) {
            std::uint8_t* row = sbuf_.at(op.dst_addr + static_cast<std::size_t>(r) * op.dst_stride,
                                         static_cast<std::size_t>(op.cols) * esize);
            for (int c = 0; c < op.cols; ++c) {
              std::int32_t x = src[r * op.psum_stride + c];
              if (op.relu) x = std::max(x, 0);
              if (op.out_int8) {
                row[c] = static_cast<std::uint8_t>(requantize_int8(x, op.shift));
              } else {
                const auto u = static_cast<std::uint32_t>(x);
                for (int b = 0; b < 4; ++b) row[c * 4 + b] = static_cast<std::uint8_t>(u >> (8 * b));
              }
            }
          }
          const std::int64_t elems = static_cast<std::int64_t>(op.rows) * op.cols;
          return cfg_.act_latency_cycles + ceil_div(elems, cfg_.act_lanes);
        }
      },
      in.op);
}

SimStats Simulator::run(const Program& program) {
  const int nsem = program.num_semaphores();
  std::array<std::vector<int>, kNumEngines> queues;
  for (std::size_t i = 0; i < program.instrs.size(); ++i) {
    const Instruction& in = program.instrs[i];
    const int idx = static_cast<int>(i);
    if (!engine_accepts(in.engine, in.op)) {
      throw std::invalid_argument("instruction " + describe(program, idx) + " has an op this engine cannot run");
    }
    for (const Wait& w : in.waits) {
      if (w.sem < 0 || w.sem >= nsem) throw std::invalid_argument("bad semaphore in " + describe(program, idx));
    }
    if (in.signal >= nsem) throw std::invalid_argument("bad signal in " + describe(program, idx));
    queues[static_cast<int>(in.engine)].push_back(idx);
  }

  struct Running {
    int idx = -1;
    std::int64_t start = 0;
    std::int64_t end = 0;
    std::vector<Access> footprint;
  };

  SimStats stats;
  trace_.clear();
  std::vector<std::int64_t> sem(static_cast<std::size_t>(nsem), 0);
  std::array<Running, kNumEngines> running{};
  std::array<std::size_t, kNumEngines> head{};
  std::int64_t now = 0;

  auto ready = [&](const Instruction& in) {
    return std::all_of(in.waits.begin(), in.waits.end(),
                       [&](const Wait& w) { return sem[static_cast<std::size_t>(w.sem)] >= w.value; });
  };

  for (;;) {
    // Issue every instruction that can start at `now`.
    bool issued = true;
    while (issued) {
      issued = false;
      for (int e = 0; e < kNumEngines; ++e) {
        if (running[e].idx != -1 || head[e] >= queues[e].size()) continue;
        const int idx = queues[e][head[e]];
        const Instruction& in = program.instrs[static_cast<std::size_t>(idx)];
        if (!ready(in)) continue;

        std::vector<Access> fp = footprint(in);
        if (cfg_.detect_races) {
          for (int o = 0; o < kNumEngines; ++o) {
            if (o == e || running[o].idx == -1) continue;
            for (const Access& x : fp) {
              for (const Access& y : running[o].footprint) {
                if (conflicts(x, y)) {
                  throw RaceError("race at cycle " + std::to_string(now) + ": " + describe(program, idx) +
                                  " overlaps in-flight " + describe(program, running[o].idx));
                }
              }
            }
          }
        }
        const std::int64_t dur = std::max<std::int64_t>(1, execute(in, stats));
        running[e] = Running{idx, now, now + dur, std::move(fp)};
        stats.engines[e].busy_cycles += dur;
        stats.engines[e].instructions += 1;
        ++head[e];
        issued = true;
      }
    }

    std::int64_t next = std::numeric_limits<std::int64_t>::max();
    for (const Running& r : running) {
      if (r.idx != -1) next = std::min(next, r.end);
    }
    if (next == std::numeric_limits<std::int64_t>::max()) {
      bool pending = false;
      std::ostringstream msg;
      msg << "deadlock at cycle " << now << ":";
      for (int e = 0; e < kNumEngines; ++e) {
        if (head[e] >= queues[e].size()) continue;
        pending = true;
        const int idx = queues[e][head[e]];
        msg << " [" << describe(program, idx) << " waits";
        for (const Wait& w : program.instrs[static_cast<std::size_t>(idx)].waits) {
          msg << ' ' << program.semaphore_names[static_cast<std::size_t>(w.sem)] << ">=" << w.value
              << " (is " << sem[static_cast<std::size_t>(w.sem)] << ')';
        }
        msg << ']';
      }
      if (pending) throw DeadlockError(msg.str());
      break;
    }

    now = next;
    for (int e = 0; e < kNumEngines; ++e) {
      Running& r = running[e];
      if (r.idx == -1 || r.end != now) continue;
      const Instruction& in = program.instrs[static_cast<std::size_t>(r.idx)];
      trace_.push_back(TraceEvent{in.engine, in.label, r.start, r.end});
      if (in.signal >= 0) ++sem[static_cast<std::size_t>(in.signal)];
      r = Running{};
    }
  }

  stats.total_cycles = now;
  return stats;
}

void Simulator::write_chrome_trace(std::ostream& os) const {
  os << "{\"traceEvents\":[\n";
  for (int e = 0; e < kNumEngines; ++e) {
    if (e > 0) os << ",\n";
    os << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":" << e << ",\"args\":{\"name\":\""
       << engine_name(static_cast<Engine>(e)) << "\"}}";
  }
  for (const TraceEvent& ev : trace_) {
    os << ",\n{\"name\":\"";
    json_escape(os, ev.label);
    os << "\",\"cat\":\"" << engine_name(ev.engine) << "\",\"ph\":\"X\",\"pid\":0,\"tid\":"
       << static_cast<int>(ev.engine) << ",\"ts\":" << ev.start << ",\"dur\":" << (ev.end - ev.start) << '}';
  }
  os << "\n],\"displayTimeUnit\":\"ns\",\"otherData\":{\"time_unit\":\"cycles\"}}\n";
}

}  // namespace tessera
