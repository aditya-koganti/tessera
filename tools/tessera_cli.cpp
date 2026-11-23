// SPDX-License-Identifier: MIT
//
// tessera-run: compile a GEMM or MLP for the simulated accelerator, run it,
// check the result against the golden model and print JSON statistics.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "tessera/compiler.hpp"
#include "tessera/reference.hpp"
#include "tessera/simulator.hpp"

namespace {

using namespace tessera;

struct Args {
  int m = 64, k = 64, n = 64;
  std::vector<int> mlp;  // dimension chain K0,K1,...,KL
  int buffers = 2, tile_m = 0;
  unsigned seed = 1;
  bool relu = false, int8_out = false;
  int shift = 0;
  HwConfig hw;
  std::string trace_path, dump_path;
};

[[noreturn]] void usage(const std::string& err) {
  if (!err.empty()) std::cerr << "error: " << err << "\n\n";
  std::cerr << R"(usage: tessera-run [options]
  workload:  --m M --k K --n N            single GEMM (default 64x64x64)
             --mlp K0,K1,...,KL           MLP with batch --m; hidden layers use ReLU + int8
             --relu --shift S --int8-out  GEMM epilogue (default: int32 output)
  compiler:  --buffers B (1..4, default 2)  --tile-m T (0 = auto)
  hardware:  --rows R --cols C --sbuf-kb KB --psum-kb KB --dma-bw BYTES --dma-latency CYCLES
             --act-lanes L --analytical (closed-form PE timing) --no-race-check
  output:    --trace FILE (Chrome/Perfetto trace)  --dump FILE (inputs/outputs as JSON)
             --seed S
)";
  std::exit(1);
}

int to_int(const std::string& flag, const char* v) {
  try {
    std::size_t pos = 0;
    const long x = std::stol(v, &pos);
    if (pos != std::strlen(v)) throw std::invalid_argument("trailing");
    return static_cast<int>(x);
  } catch (const std::exception&) {
    usage("bad integer for " + flag + ": " + v);
  }
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    auto val = [&]() -> const char* {
      if (i + 1 >= argc) usage("missing value for " + f);
      return argv[++i];
    };
    if (f == "--m") a.m = to_int(f, val());
    else if (f == "--k") a.k = to_int(f, val());
    else if (f == "--n") a.n = to_int(f, val());
    else if (f == "--mlp") {
      std::stringstream ss(val());
      std::string part;
      while (std::getline(ss, part, ',')) a.mlp.push_back(to_int(f, part.c_str()));
      if (a.mlp.size() < 2) usage("--mlp needs at least two dimensions");
    } else if (f == "--buffers") a.buffers = to_int(f, val());
    else if (f == "--tile-m") a.tile_m = to_int(f, val());
    else if (f == "--seed") a.seed = static_cast<unsigned>(to_int(f, val()));
    else if (f == "--relu") a.relu = true;
    else if (f == "--shift") a.shift = to_int(f, val());
    else if (f == "--int8-out") a.int8_out = true;
    else if (f == "--rows") a.hw.array_rows = to_int(f, val());
    else if (f == "--cols") a.hw.array_cols = to_int(f, val());
    else if (f == "--sbuf-kb") a.hw.sbuf_bytes = static_cast<std::size_t>(to_int(f, val())) * 1024;
    else if (f == "--psum-kb") a.hw.psum_bytes = static_cast<std::size_t>(to_int(f, val())) * 1024;
    else if (f == "--dma-bw") a.hw.dma_bytes_per_cycle = to_int(f, val());
    else if (f == "--dma-latency") a.hw.dma_latency_cycles = to_int(f, val());
    else if (f == "--act-lanes") a.hw.act_lanes = to_int(f, val());
    else if (f == "--analytical") a.hw.cycle_accurate_pe = false;
    else if (f == "--no-race-check") a.hw.detect_races = false;
    else if (f == "--trace") a.trace_path = val();
    else if (f == "--dump") a.dump_path = val();
    else if (f == "-h" || f == "--help") usage("");
    else usage("unknown option " + f);
  }
  return a;
}

std::size_t align_up(std::size_t x) { return (x + 63) / 64 * 64; }

template <typename T>
void write_matrix(std::ostream& os, const std::vector<T>& v, int rows, int cols) {
  os << '[';
  for (int r = 0; r < rows; ++r) {
    os << (r ? ",[" : "[");
    for (int c = 0; c < cols; ++c) {
      os << (c ? "," : "") << static_cast<long long>(v[static_cast<std::size_t>(r) * cols + c]);
    }
    os << ']';
  }
  os << ']';
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  try {
    args.hw.validate();

    // Describe the workload as a chain of layers.
    std::vector<int> dims = args.mlp.empty() ? std::vector<int>{args.k, args.n} : args.mlp;
    const int M = args.m;
    const std::size_t nl = dims.size() - 1;
    std::vector<GemmLayer> layers(nl);
    std::size_t next = 0;
    const std::size_t a0 = next;
    next = align_up(next + static_cast<std::size_t>(M) * dims[0]);
    for (std::size_t l = 0; l < nl; ++l) {
      GemmLayer& L = layers[l];
      L.m = M;
      L.k = dims[l];
      L.n = dims[l + 1];
      const bool last = l + 1 == nl;
      if (args.mlp.empty()) {
        L.relu = args.relu;
        L.shift = args.shift;
        L.out_int8 = args.int8_out;
      } else {
        L.relu = !last;
        L.out_int8 = !last;
        L.shift = last ? 0 : (args.shift > 0 ? args.shift : 8);
      }
      L.a_dram = l == 0 ? a0 : layers[l - 1].c_dram;
      L.w_dram = next;
      next = align_up(next + static_cast<std::size_t>(L.k) * L.n);
      L.c_dram = next;
      next = align_up(next + static_cast<std::size_t>(M) * L.n * (L.out_int8 ? 1 : 4));
    }
    if (next > args.hw.dram_bytes) throw std::invalid_argument("workload does not fit in DRAM");

    // Random inputs.
    std::mt19937 rng(args.seed);
    std::uniform_int_distribution<int> dist(-128, 127);
    auto rand_vec = [&](std::size_t count) {
      std::vector<std::int8_t> v(count);
      for (auto& x : v) x = static_cast<std::int8_t>(dist(rng));
      return v;
    };
    const std::vector<std::int8_t> a = rand_vec(static_cast<std::size_t>(M) * dims[0]);
    std::vector<std::vector<std::int8_t>> weights;
    for (const GemmLayer& L : layers) weights.push_back(rand_vec(static_cast<std::size_t>(L.k) * L.n));

    CompileOptions opts;
    opts.buffers = args.buffers;
    opts.tile_m = args.tile_m;
    std::vector<LayerPlan> plans;
    const Program prog = compile(args.hw, layers, opts, &plans);

    Simulator sim(args.hw);
    std::memcpy(sim.dram().at(layers[0].a_dram, a.size()), a.data(), a.size());
    for (std::size_t l = 0; l < nl; ++l) {
      std::memcpy(sim.dram().at(layers[l].w_dram, weights[l].size()), weights[l].data(), weights[l].size());
    }
    const SimStats stats = sim.run(prog);

    // Golden model.
    std::vector<std::int8_t> x = a;
    std::vector<std::int32_t> expected;
    std::vector<std::int8_t> expected8;
    for (std::size_t l = 0; l < nl; ++l) {
      const GemmLayer& L = layers[l];
      const std::vector<std::int32_t> acc = gemm_ref(x, weights[l], L.m, L.k, L.n);
      if (L.out_int8) {
        expected8 = epilogue_int8(acc, L.relu, L.shift);
        x = expected8;
      } else {
        expected = epilogue_int32(acc, L.relu);
      }
    }
    const GemmLayer& out = layers.back();
    const std::size_t count = static_cast<std::size_t>(out.m) * out.n;
    std::vector<std::int32_t> got(count);
    long long mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (out.out_int8) {
        got[i] = static_cast<std::int8_t>(*sim.dram().at(out.c_dram + i, 1));
        if (got[i] != expected8[i]) ++mismatches;
      } else {
        const std::uint8_t* p = sim.dram().at(out.c_dram + 4 * i, 4);
        const std::uint32_t u = p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
        got[i] = static_cast<std::int32_t>(u);
        if (got[i] != expected[i]) ++mismatches;
      }
    }

    if (!args.trace_path.empty()) {
      std::ofstream f(args.trace_path);
      sim.write_chrome_trace(f);
    }
    if (!args.dump_path.empty()) {
      std::ofstream f(args.dump_path);
      f << "{\"m\":" << M << ",\"layers\":[";
      for (std::size_t l = 0; l < nl; ++l) {
        const GemmLayer& L = layers[l];
        f << (l ? "," : "") << "{\"k\":" << L.k << ",\"n\":" << L.n << ",\"relu\":" << (L.relu ? "true" : "false")
          << ",\"shift\":" << L.shift << ",\"out_int8\":" << (L.out_int8 ? "true" : "false") << ",\"w\":";
        write_matrix(f, weights[l], L.k, L.n);
        f << '}';
      }
      f << "],\"a\":";
      write_matrix(f, a, M, dims[0]);
      f << ",\"output\":";
      write_matrix(f, got, out.m, out.n);
      f << "}\n";
    }

    const HwConfig& hw = args.hw;
    std::cout << std::setprecision(6) << std::fixed;
    std::cout << "{\n  \"config\": {\"rows\": " << hw.array_rows << ", \"cols\": " << hw.array_cols
              << ", \"buffers\": " << args.buffers << ", \"sbuf_bytes\": " << hw.sbuf_bytes
              << ", \"psum_bytes\": " << hw.psum_bytes << ", \"dma_bytes_per_cycle\": " << hw.dma_bytes_per_cycle
              << ", \"dma_latency_cycles\": " << hw.dma_latency_cycles
              << ", \"cycle_accurate_pe\": " << (hw.cycle_accurate_pe ? "true" : "false") << "},\n";
    std::cout << "  \"layers\": [";
    for (std::size_t l = 0; l < nl; ++l) {
      const GemmLayer& L = layers[l];
      const LayerPlan& p = plans[l];
      std::cout << (l ? ", " : "") << "{\"m\": " << L.m << ", \"k\": " << L.k << ", \"n\": " << L.n
                << ", \"tile_m\": " << p.tile_m << ", \"tiles\": [" << p.tiles_m << ", " << p.tiles_k << ", "
                << p.tiles_n << "], \"matmuls\": " << p.matmuls << ", \"sbuf_bytes_used\": " << p.sbuf_bytes_used
                << ", \"psum_bytes_used\": " << p.psum_bytes_used << "}";
    }
    std::cout << "],\n  \"instructions\": " << prog.instrs.size() << ",\n  \"cycles\": " << stats.total_cycles
              << ",\n  \"macs\": " << stats.useful_macs << ",\n  \"pe_utilization\": " << stats.pe_utilization(hw)
              << ",\n  \"engines\": {";
    for (int e = 0; e < kNumEngines; ++e) {
      const auto eng = static_cast<Engine>(e);
      std::cout << (e ? ", " : "") << '"' << engine_name(eng)
                << "\": {\"busy_cycles\": " << stats.engines[e].busy_cycles
                << ", \"instructions\": " << stats.engines[e].instructions
                << ", \"occupancy\": " << stats.occupancy(eng) << '}';
    }
    std::cout << "},\n  \"dma_bytes\": {\"in\": " << stats.dma_bytes_in << ", \"out\": " << stats.dma_bytes_out
              << "},\n  \"correct\": " << (mismatches == 0 ? "true" : "false") << ",\n  \"mismatches\": "
              << mismatches << "\n}\n";
    return mismatches == 0 ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "{\"error\": \"" << e.what() << "\"}\n";
    return 1;
  }
}
