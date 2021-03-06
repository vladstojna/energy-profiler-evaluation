#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>

#include <cassert>
#include <charconv>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <omp.h>

namespace {
tp::printer g_tpr;

struct cmdargs {
  std::size_t count = 0;
  double lower = 0.0;
  double upper = 1.0;

  cmdargs(int argc, const char *const *argv) {
    if (argc < 2) {
      print_usage(argv[0]);
      throw std::invalid_argument("Not enough arguments");
    }
    util::to_scalar(argv[1], count);
    if (argc > 2)
      util::to_scalar(argv[2], lower);
    if (argc > 3)
      util::to_scalar(argv[3], upper);
    if (lower >= upper) {
      print_usage(argv[0]);
      throw std::invalid_argument("<lower> must be less than <upper>");
    }
  }

private:
  void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog << " <count> <lower> <upper>\n";
  }
};

__attribute__((noinline)) void generate(std::vector<std::mt19937_64> &engines,
                                        std::size_t count, double lower,
                                        double upper) {
#pragma omp parallel
  {
    auto &engine = engines[omp_get_thread_num()];
    std::uniform_real_distribution dist{lower, upper};
#pragma omp for
    for (std::size_t i = 0; i < count; i++)
      std::ignore = dist(engine);
  }
}

std::vector<std::mt19937_64> get_engines() {
  std::random_device rd;
  std::vector<std::mt19937_64> engines;

  int max_threads = omp_get_max_threads();
  engines.reserve(max_threads);
  for (int i = 0; i < max_threads; i++) {
    std::seed_seq sseq{rd(), rd(), rd()};
    engines.emplace_back(sseq);
  }
  return engines;
}
} // namespace

int main(int argc, char **argv) {
  try {
    const cmdargs args(argc, argv);
    std::vector<std::mt19937_64> engines = get_engines();
    tp::sampler s(g_tpr);
    generate(engines, args.count, args.lower, args.upper);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
}
