#if defined V_USE_OPENBLAS
#include <cblas.h>
#elif defined V_USE_MKL
#include <mkl.h>
#endif

#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

#define NO_INLINE __attribute__((noinline))

namespace {
tp::printer g_tpr;

namespace detail {
template <auto Func>
struct func_obj : std::integral_constant<decltype(Func), Func> {};

template <typename> struct gemm_caller {};

template <> struct gemm_caller<float> : func_obj<cblas_sgemm> {};

template <> struct gemm_caller<double> : func_obj<cblas_dgemm> {};

template <typename> struct gemv_caller {};

template <> struct gemv_caller<float> : func_obj<cblas_sgemv> {};

template <> struct gemv_caller<double> : func_obj<cblas_dgemv> {};

struct transpose : std::integral_constant<decltype(CblasTrans), CblasTrans> {};
struct no_transpose
    : std::integral_constant<decltype(CblasNoTrans), CblasNoTrans> {};

template <typename Transpose, typename Real>
NO_INLINE void gemm_compute(std::size_t iters, std::size_t M, std::size_t N,
                            std::size_t K, const Real *a, const Real *b,
                            Real *c) {
  tp::sampler smp(g_tpr);
  for (decltype(iters) i = 0; i < iters; i++)
    gemm_caller<Real>::value(CblasRowMajor, CblasNoTrans, Transpose::value, M,
                             N, K, 1.0, a, K, b, N, 0, c, N);
}

template <typename Real, typename Transpose>
void gemm_impl(std::size_t M, std::size_t N, std::size_t K, std::size_t iters,
               std::mt19937_64 &engine) {
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  auto gen = [&]() { return dist(engine); };

  tp::sampler smp(g_tpr);
  std::vector<Real> a(M * K);
  std::vector<Real> b(K * N);
  std::vector<Real> c(M * N);
  std::generate(a.begin(), a.end(), gen);
  std::generate(b.begin(), b.end(), gen);
  gemm_compute<Transpose>(iters, M, N, K, a.data(), b.data(), c.data());
}

template <typename Real>
NO_INLINE void gemv_compute(std::size_t iters, std::size_t M, std::size_t N,
                            const Real *a, const Real *x, Real *y) {
  tp::sampler smp(g_tpr);
  for (decltype(iters) i = 0; i < iters; i++)
    gemv_caller<Real>::value(CblasRowMajor, CblasNoTrans, M, N, 1.0, a, N, x, 1,
                             0, y, 1);
}

template <typename Real>
void gemv_impl(std::size_t M, std::size_t N, std::size_t iters,
               std::mt19937_64 &engine) {
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  auto gen = [&]() { return dist(engine); };

  tp::sampler smp(g_tpr);
  std::vector<Real> a(M * N);
  std::vector<Real> x(N);
  std::vector<Real> y(M);
  std::generate(a.begin(), a.end(), gen);
  gemv_compute(iters, M, N, a.data(), x.data(), y.data());
}
} // namespace detail

#define DECLARE_FUNC(f)                                                        \
  NO_INLINE void f(std::size_t, std::size_t, std::size_t, std::size_t,         \
                   std::mt19937_64 &)

DECLARE_FUNC(dgemm_notrans);
DECLARE_FUNC(dgemm);
DECLARE_FUNC(sgemm_notrans);
DECLARE_FUNC(sgemm);
DECLARE_FUNC(dgemv);
DECLARE_FUNC(sgemv);

void dgemm_notrans(std::size_t M, std::size_t N, std::size_t K,
                   std::size_t iters, std::mt19937_64 &engine) {
  detail::gemm_impl<double, detail::no_transpose>(M, N, K, iters, engine);
}

void dgemm(std::size_t M, std::size_t N, std::size_t K, std::size_t iters,
           std::mt19937_64 &engine) {
  detail::gemm_impl<double, detail::transpose>(M, N, K, iters, engine);
}

void sgemm_notrans(std::size_t M, std::size_t N, std::size_t K,
                   std::size_t iters, std::mt19937_64 &engine) {
  detail::gemm_impl<float, detail::no_transpose>(M, N, K, iters, engine);
}

void sgemm(std::size_t M, std::size_t N, std::size_t K, std::size_t iters,
           std::mt19937_64 &engine) {
  detail::gemm_impl<float, detail::transpose>(M, N, K, iters, engine);
}

void dgemv(std::size_t M, std::size_t N, std::size_t, std::size_t iters,
           std::mt19937_64 &engine) {
  detail::gemv_impl<double>(M, N, iters, engine);
}

void sgemv(std::size_t M, std::size_t N, std::size_t, std::size_t iters,
           std::mt19937_64 &engine) {
  detail::gemv_impl<float>(M, N, iters, engine);
}

struct cmdparams {
  using work_func = decltype(&sgemm);
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  std::size_t iters = 1;
  work_func func = nullptr;

  cmdparams(int argc, const char *const *argv) {
    if (argc < 4) {
      print_usage(argv[0]);
      throw std::invalid_argument("Not enough arguments");
    }
    std::string op_type = argv[1];
    std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    util::to_scalar(argv[2], m);
    assert_positive(m, "m");
    util::to_scalar(argv[3], n);
    assert_positive(n, "n");
    if (op_type == "dgemv") {
      func = dgemv;
      get_iterations(argc, argv, 4);
    } else if (op_type == "sgemv") {
      func = sgemv;
      get_iterations(argc, argv, 4);
    } else {
      if (argc < 5) {
        print_usage(argv[0]);
        throw std::invalid_argument(op_type.append(": not enough arguments"));
      }
      util::to_scalar(argv[4], k);
      assert_positive(k, "k");
      if (op_type == "dgemm")
        func = dgemm;
      else if (op_type == "dgemm_notrans")
        func = dgemm_notrans;
      else if (op_type == "sgemm")
        func = sgemm;
      else if (op_type == "sgemm_notrans")
        func = sgemm_notrans;
      else {
        print_usage(argv[0]);
        throw std::invalid_argument(
            std::string("invalid work type: ").append(op_type));
      }
      get_iterations(argc, argv, 5);
    }
    assert(func);
  }

  void do_work(std::mt19937_64 &engine) const { func(m, n, k, iters, engine); }

private:
  void assert_positive(std::size_t x, std::string name) {
    assert(x);
    if (!x)
      throw std::invalid_argument(
          std::move(name.append(" must be greater than 0")));
  }

  void get_iterations(int argc, const char *const *argv, int idx) {
    if (argc > idx)
      util::to_scalar(argv[idx], iters);
    assert_positive(iters, "iters");
  }

  void print_usage(const char *prog) {
    std::cerr
        << "Usage:\n"
        << "\t" << prog
        << " {dgemm,dgemm_notrans,sgemm,sgemm_notrans} <m> <n> <k> <iters>\n"
        << "\t" << prog << " {dgemv,sgemv} <m> <n> <iters>\n";
  }
};
} // namespace

int main(int argc, char **argv) {
  try {
    const cmdparams params(argc, argv);
    std::random_device rnd_dev;
    std::mt19937_64 engine{rnd_dev()};
    params.do_work(engine);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
