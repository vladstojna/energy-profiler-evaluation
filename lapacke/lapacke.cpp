#if defined V_USE_OPENBLAS
#include <lapacke.h>
#elif defined V_USE_MKL
#include <mkl.h>
#endif

#include <timeprinter/printer.hpp>
#include <util/buffer.hpp>
#include <util/to_scalar.hpp>

#include <algorithm>
#include <cassert>
#include <random>

#define NO_INLINE __attribute__((noinline))

#if defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)
#define NO_CLONE __attribute__((noclone, no_icf))
#else // !defined(USE_ITERATIONS) || defined(DO_COMPUTATION)
#define NO_CLONE __attribute__((noclone))
#endif // defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)

namespace {
tp::printer g_tpr;

#if defined(USE_ITERATIONS)
std::size_t g_iters = 0;
#endif // defined(USE_ITERATIONS)

struct compute_params {
  std::size_t N = 0;
  std::size_t M = 0;
  std::size_t Nrhs = 0;
};

namespace detail {
#if defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)
void handle_error(int) {}
#else  // !defined(USE_ITERATIONS) || defined(DO_COMPUTATION)
void handle_error(int res) {
  if (res > 0)
    throw std::runtime_error(
        std::string("The solution could not be computed; info=")
            .append(std::to_string(res)));
  if (res < 0)
    throw std::runtime_error(std::string("Error during computation; info=")
                                 .append(std::to_string(res)));
}
#endif // defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)

#if defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)
#define DEFINE_ALIAS(prefix)                                                   \
  template <typename Real> using prefix##_caller = any_caller<Real>

template <typename Real> struct any_caller {
  struct impl {
    template <typename... Args> constexpr int operator()(Args &&...) const {
      return 0;
    }
  };
  using type = Real;
  static constexpr auto value = impl{};
};

DEFINE_ALIAS(gesv);
DEFINE_ALIAS(gels);
DEFINE_ALIAS(getri);
DEFINE_ALIAS(getrf);
DEFINE_ALIAS(getrs);
DEFINE_ALIAS(tptri);
DEFINE_ALIAS(trtri);
DEFINE_ALIAS(potrf);
#else // !defined(USE_ITERATIONS) || defined(DO_COMPUTATION)
#define DEFINE_CALLER(prefix)                                                  \
  template <typename> struct prefix##_caller {};                               \
  template <> struct prefix##_caller<float> {                                  \
    static constexpr auto value = LAPACKE_s##prefix##_work;                    \
    using type = float;                                                        \
  };                                                                           \
  template <> struct prefix##_caller<double> {                                 \
    static constexpr auto value = LAPACKE_d##prefix##_work;                    \
    using type = double;                                                       \
  }

DEFINE_CALLER(gesv);
DEFINE_CALLER(gels);
DEFINE_CALLER(getri);
DEFINE_CALLER(getrf);
DEFINE_CALLER(getrs);
DEFINE_CALLER(tptri);
DEFINE_CALLER(trtri);
DEFINE_CALLER(potrf);
#endif // defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)

// upper triangular in row-major is lower triangular in column-major,
// therefore pass 'L' to function which expects a column-major format
template <typename It, typename Gen>
void fill_upper_triangular(It from, It to, std::size_t ld, Gen gen) {
  for (auto [it, nnz] = std::pair{from, ld}; it < to; it += ld + 1, nnz--)
    for (auto entry = it; entry < it + nnz; entry++)
      *entry = gen();
}

template <typename Real>
util::buffer<Real> upper_dd_matrix(std::size_t N, std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  auto gen = [&]() { return dist(engine); };

  util::buffer<Real> a{N * N};
  std::fill(a.begin(), a.end(), Real{});
  fill_upper_triangular(a.begin(), a.end(), N, gen);
  smp.do_sample();
  {
    // compute A = A + rand(N, 2N) * Identity(N, N)
    // to guarantee that the matrix is diagonally dominant
    std::uniform_real_distribution<Real> dist{static_cast<Real>(N),
                                              static_cast<Real>(2 * N)};
    for (auto [it, x] = std::pair{a.begin(), 0}; it < a.end(); it += N, ++x)
      *(it + x) += dist(engine);
  }
  return a;
}

namespace compute {
namespace impl {
template <typename Caller, typename... Args> void generic_solver(Args... args) {
  handle_error(Caller::value(args...));
}

template <typename Caller, typename... Args> void query_solver(Args... args) {
  using Real = typename Caller::type;
  auto workspace_query = Real{};
  handle_error(Caller::value(args..., &workspace_query, -1));
  util::buffer<Real> work{static_cast<std::uint64_t>(workspace_query)};
  handle_error(Caller::value(args..., work.get(), work.size()));
}

#if defined(USE_ITERATIONS)
template <typename Real>
void gesv(Real *a, Real *b, lapack_int *ipiv, size_t N, size_t Nrhs) {
  util::buffer<Real> a2{N * N};
  util::buffer<Real> b2{N * Nrhs};
  for (size_t i = 0; i < g_iters; i++) {
    std::ignore = std::copy(a, a + a2.size(), std::begin(a2));
    std::ignore = std::copy(b, b + b2.size(), std::begin(b2));
    generic_solver<gesv_caller<Real>>(LAPACK_COL_MAJOR, N, Nrhs, a2.get(), N,
                                      ipiv, b2.get(), N);
  }
}

template <typename Real>
void gels(Real *a, Real *b, size_t M, size_t N, size_t Nrhs) {
  util::buffer<Real> a2{M * N};
  util::buffer<Real> b2{std::max(N, M) * Nrhs};
  for (size_t i = 0; i < g_iters; i++) {
    std::ignore = std::copy(a, a + a2.size(), std::begin(a2));
    std::ignore = std::copy(b, b + b2.size(), std::begin(b2));
    query_solver<gels_caller<Real>>(LAPACK_COL_MAJOR, 'N', M, N, Nrhs, a2.get(),
                                    M, b2.get(), std::max(N, M));
  }
}

template <typename Real>
void getrf(Real *a, lapack_int *ipiv, size_t M, size_t N) {
  util::buffer<Real> a2{M * N};
  for (size_t i = 0; i < g_iters; i++) {
    std::ignore = std::copy(a, a + a2.size(), std::begin(a2));
    generic_solver<getrf_caller<Real>>(LAPACK_COL_MAJOR, M, N, a2.get(), M,
                                       ipiv);
  }
}

template <typename Real> void getri(Real *a, const lapack_int *ipiv, size_t N) {
  util::buffer<Real> a2{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    std::ignore = std::copy(a, a + a2.size(), std::begin(a2));
    query_solver<getri_caller<Real>>(LAPACK_COL_MAJOR, N, a2.get(), N, ipiv);
  }
}

template <typename Real>
void getrs(const Real *a, Real *b, const lapack_int *ipiv, size_t N,
           size_t Nrhs) {
  util::buffer<Real> b2{N * Nrhs};
  for (size_t i = 0; i < g_iters; i++) {
    std::ignore = std::copy(b, b + b2.size(), std::begin(b2));
    generic_solver<getrs_caller<Real>>(LAPACK_COL_MAJOR, 'N', N, Nrhs, a, N,
                                       ipiv, b2.get(), N);
  }
}

template <typename Real> void tptri(Real *a, size_t N) {
  util::buffer<Real> a2{N * (N + 1) / 2};
  for (size_t i = 0; i < g_iters; i++) {
    std::ignore = std::copy(a, a + a2.size(), std::begin(a2));
    generic_solver<tptri_caller<Real>>(LAPACK_COL_MAJOR, 'L', 'N', N, a2.get());
  }
}

template <typename Real> void trtri(Real *a, size_t N) {
  util::buffer<Real> a2{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    std::ignore = std::copy(a, a + a2.size(), std::begin(a2));
    generic_solver<trtri_caller<Real>>(LAPACK_COL_MAJOR, 'L', 'N', N, a2.get(),
                                       N);
  }
}

template <typename Real> void potrf(Real *a, size_t N) {
  util::buffer<Real> a2{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    std::ignore = std::copy(a, a + a2.size(), std::begin(a2));
    generic_solver<potrf_caller<Real>>(LAPACK_COL_MAJOR, 'L', N, a2.get(), N);
  }
}
#else  // !defined(USE_ITERATIONS)
template <typename Real>
void gesv(Real *a, Real *b, lapack_int *ipiv, size_t N, size_t Nrhs) {
  generic_solver<gesv_caller<Real>>(LAPACK_COL_MAJOR, N, Nrhs, a, N, ipiv, b,
                                    N);
}

template <typename Real>
void gels(Real *a, Real *b, size_t M, size_t N, size_t Nrhs) {
  query_solver<gels_caller<Real>>(LAPACK_COL_MAJOR, 'N', M, N, Nrhs, a, M, b,
                                  std::max(N, M));
}

template <typename Real>
void getrf(Real *a, lapack_int *ipiv, size_t M, size_t N) {
  generic_solver<getrf_caller<Real>>(LAPACK_COL_MAJOR, M, N, a, M, ipiv);
}

template <typename Real> void getri(Real *a, const lapack_int *ipiv, size_t N) {
  query_solver<getri_caller<Real>>(LAPACK_COL_MAJOR, N, a, N, ipiv);
}

template <typename Real>
void getrs(const Real *a, Real *b, const lapack_int *ipiv, size_t N,
           size_t Nrhs) {
  generic_solver<getrs_caller<Real>>(LAPACK_COL_MAJOR, 'N', N, Nrhs, a, N, ipiv,
                                     b, N);
}

template <typename Real> void tptri(Real *a, size_t N) {
  generic_solver<tptri_caller<Real>>(LAPACK_COL_MAJOR, 'L', 'N', N, a);
}

template <typename Real> void trtri(Real *a, size_t N) {
  generic_solver<trtri_caller<Real>>(LAPACK_COL_MAJOR, 'L', 'N', N, a, N);
}

template <typename Real> void potrf(Real *a, size_t N) {
  generic_solver<potrf_caller<Real>>(LAPACK_COL_MAJOR, 'L', N, a, N);
}
#endif // defined(USE_ITERATIONS)
} // namespace impl

template <typename Real>
NO_INLINE NO_CLONE void gesv(Real *a, Real *b, lapack_int *ipiv, std::size_t N,
                             std::size_t Nrhs) {
  tp::sampler smp(g_tpr);
  impl::gesv(a, b, ipiv, N, Nrhs);
}

template <typename Real>
NO_INLINE NO_CLONE void gels(Real *a, Real *b, std::size_t M, std::size_t N,
                             std::size_t Nrhs) {
  tp::sampler smp(g_tpr);
  impl::gels(a, b, M, N, Nrhs);
}

template <typename Real>
NO_INLINE NO_CLONE void getrf(Real *a, lapack_int *ipiv, std::size_t M,
                              std::size_t N) {
  tp::sampler smp(g_tpr);
  impl::getrf(a, ipiv, M, N);
}

template <typename Real>
NO_INLINE NO_CLONE void getri(Real *a, const lapack_int *ipiv, std::size_t N) {
  tp::sampler smp(g_tpr);
  impl::getri(a, ipiv, N);
}

template <typename Real>
NO_INLINE NO_CLONE void getrs(const Real *a, Real *b, const lapack_int *ipiv,
                              std::size_t N, std::size_t Nrhs) {
  tp::sampler smp(g_tpr);
  impl::getrs(a, b, ipiv, N, Nrhs);
}

template <typename Real> NO_INLINE NO_CLONE void tptri(Real *a, std::size_t N) {
  tp::sampler smp(g_tpr);
  impl::tptri(a, N);
}

template <typename Real> NO_INLINE NO_CLONE void trtri(Real *a, std::size_t N) {
  tp::sampler smp(g_tpr);
  impl::trtri(a, N);
}

template <typename Real> NO_INLINE NO_CLONE void potrf(Real *a, std::size_t N) {
  tp::sampler smp(g_tpr);
  impl::potrf(a, N);
}
} // namespace compute

template <typename Real>
void gesv_impl(std::size_t N, std::size_t Nrhs, std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  util::buffer<Real> a{N * N};
  util::buffer<Real> b{N * Nrhs};
  util::buffer<lapack_int> ipiv{N};
  auto gen = [&]() { return dist(engine); };
  std::generate(a.begin(), a.end(), gen);
  std::generate(b.begin(), b.end(), gen);
  compute::gesv(a.get(), b.get(), ipiv.get(), N, Nrhs);
}

template <typename Real>
void gels_impl(std::size_t M, std::size_t N, std::size_t Nrhs,
               std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  util::buffer<Real> a{M * N};
  util::buffer<Real> b{std::max(N, M) * Nrhs};
  auto gen = [&]() { return dist(engine); };
  std::generate(a.begin(), a.end(), gen);
  std::generate(b.begin(), b.end(), gen);
  compute::gels(a.get(), b.get(), M, N, Nrhs);
}

template <typename Real>
void getri_impl(std::size_t N, std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  util::buffer<Real> a{N * N};
  util::buffer<lapack_int> ipiv{N};
  std::generate(a.begin(), a.end(), [&]() { return dist(engine); });
  compute::getrf(a.get(), ipiv.get(), N, N);
  compute::getri(a.get(), ipiv.get(), N);
}

template <typename Real>
void getrf_impl(std::size_t M, std::size_t N, std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  util::buffer<Real> a{M * N};
  util::buffer<lapack_int> ipiv{std::min(M, N)};
  auto gen = [&]() { return dist(engine); };
  std::generate(a.begin(), a.end(), gen);
  compute::getrf(a.get(), ipiv.get(), M, N);
}

template <typename Real>
void getrs_impl(std::size_t N, std::size_t Nrhs, std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  util::buffer<Real> a{N * N};
  util::buffer<Real> b{N * Nrhs};
  util::buffer<lapack_int> ipiv{N};
  auto gen = [&]() { return dist(engine); };
  std::generate(a.begin(), a.end(), gen);
  std::generate(b.begin(), b.end(), gen);
  compute::getrf(a.get(), ipiv.get(), N, N);
  compute::getrs(a.get(), b.get(), ipiv.get(), N, Nrhs);
}

template <typename Real>
void tptri_impl(std::size_t N, std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{1.0, 2.0};
  util::buffer<Real> a_packed{N * (N + 1) / 2};
  auto gen = [&]() { return dist(engine); };
  std::generate(a_packed.begin(), a_packed.end(), gen);
  compute::tptri(a_packed.get(), N);
}

template <typename Real>
void trtri_impl(std::size_t N, std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{1.0, 2.0};
  util::buffer<Real> a{N * N};
  auto gen = [&]() { return dist(engine); };
  fill_upper_triangular(a.begin(), a.end(), N, gen);
  compute::trtri(a.get(), N);
}

template <typename Real>
void potrf_impl(std::size_t N, std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  auto a = upper_dd_matrix<Real>(N, engine);
  compute::potrf(a.get(), N);
}
} // namespace detail

NO_INLINE void dgesv(compute_params p, std::mt19937_64 &engine) {
  detail::gesv_impl<double>(p.N, p.Nrhs, engine);
}

NO_INLINE void sgesv(compute_params p, std::mt19937_64 &engine) {
  detail::gesv_impl<float>(p.N, p.Nrhs, engine);
}

NO_INLINE void dgetrs(compute_params p, std::mt19937_64 &engine) {
  detail::getrs_impl<double>(p.N, p.Nrhs, engine);
}

NO_INLINE void sgetrs(compute_params p, std::mt19937_64 &engine) {
  detail::getrs_impl<float>(p.N, p.Nrhs, engine);
}

NO_INLINE void dgels(compute_params p, std::mt19937_64 &engine) {
  detail::gels_impl<double>(p.M, p.N, p.Nrhs, engine);
}

NO_INLINE void sgels(compute_params p, std::mt19937_64 &engine) {
  detail::gels_impl<float>(p.M, p.N, p.Nrhs, engine);
}

NO_INLINE void dgetri(compute_params p, std::mt19937_64 &engine) {
  detail::getri_impl<double>(p.N, engine);
}

NO_INLINE void sgetri(compute_params p, std::mt19937_64 &engine) {
  detail::getri_impl<float>(p.N, engine);
}

NO_INLINE void dgetrf(compute_params p, std::mt19937_64 &engine) {
  detail::getrf_impl<double>(p.M, p.N, engine);
}

NO_INLINE void sgetrf(compute_params p, std::mt19937_64 &engine) {
  detail::getrf_impl<float>(p.M, p.N, engine);
}

NO_INLINE void dtptri(compute_params p, std::mt19937_64 &engine) {
  detail::tptri_impl<double>(p.N, engine);
}

NO_INLINE void stptri(compute_params p, std::mt19937_64 &engine) {
  detail::tptri_impl<float>(p.N, engine);
}

NO_INLINE void dtrtri(compute_params p, std::mt19937_64 &engine) {
  detail::trtri_impl<double>(p.N, engine);
}

NO_INLINE void strtri(compute_params p, std::mt19937_64 &engine) {
  detail::trtri_impl<float>(p.N, engine);
}

NO_INLINE void dpotrf(compute_params p, std::mt19937_64 &engine) {
  detail::potrf_impl<double>(p.N, engine);
}

NO_INLINE void spotrf(compute_params p, std::mt19937_64 &engine) {
  detail::potrf_impl<float>(p.N, engine);
}

class cmdparams {
  using work_func = decltype(&dgesv);
  work_func func = nullptr;
  compute_params params = {};

public:
  cmdparams(int argc, const char *const *argv) {
    if (argc < 2) {
      print_usage(argv[0]);
      throw std::invalid_argument("Too few arguments");
    }
    std::string op_type = argv[1];
    std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    func = get_work_func(op_type);
    if (func == dgels || func == sgels) {
      if (argc < 5)
        too_few(argv[0], std::move(op_type));
      util::to_scalar(argv[2], params.M);
      assert_positive(params.M, "m");
      util::to_scalar(argv[3], params.N);
      assert_positive(params.N, "n");
      util::to_scalar(argv[4], params.Nrhs);
      assert_positive(params.Nrhs, "nrhs");
      get_iterations(argc, argv, 5);
    } else if (func == dgesv || func == sgesv || func == dgetrs ||
               func == sgetrs) {
      if (argc < 4)
        too_few(argv[0], std::move(op_type));
      util::to_scalar(argv[2], params.N);
      assert_positive(params.N, "n");
      util::to_scalar(argv[3], params.Nrhs);
      assert_positive(params.Nrhs, "nrhs");
      get_iterations(argc, argv, 4);
    } else if (single_arg(func)) {
      if (argc < 3)
        too_few(argv[0], std::move(op_type));
      util::to_scalar(argv[2], params.N);
      assert_positive(params.N, "n");
      get_iterations(argc, argv, 3);
    } else if (func == dgetrf || func == sgetrf) {
      if (argc < 4)
        too_few(argv[0], std::move(op_type));
      util::to_scalar(argv[2], params.M);
      assert_positive(params.M, "m");
      util::to_scalar(argv[3], params.N);
      assert_positive(params.N, "n");
      get_iterations(argc, argv, 4);
    }
  }

  void do_work(std::mt19937_64 &engine) const { func(params, engine); }

private:
  bool is_inversion(work_func func) {
    return func == dgetri || func == sgetri || func == dtptri ||
           func == stptri || func == dtrtri || func == strtri;
  }

  bool single_arg(work_func func) {
    return is_inversion(func) || func == dpotrf || func == spotrf;
  }

  void assert_positive(std::size_t x, std::string name) {
    assert(x);
    if (!x)
      throw std::invalid_argument(
          std::move(name.append(" must be greater than 0")));
  }

  work_func get_work_func(const std::string &str) {
    if (str == "dgels")
      return dgels;
    if (str == "dgesv")
      return dgesv;
    if (str == "sgels")
      return sgels;
    if (str == "sgesv")
      return sgesv;
    if (str == "dgetri")
      return dgetri;
    if (str == "sgetri")
      return sgetri;
    if (str == "dgetrf")
      return dgetrf;
    if (str == "sgetrf")
      return sgetrf;
    if (str == "dtptri")
      return dtptri;
    if (str == "stptri")
      return stptri;
    if (str == "dtrtri")
      return dtrtri;
    if (str == "strtri")
      return strtri;
    if (str == "dgetrs")
      return dgetrs;
    if (str == "sgetrs")
      return sgetrs;
    if (str == "dpotrf")
      return dpotrf;
    if (str == "spotrf")
      return spotrf;
    throw std::invalid_argument(std::string("invalid work type: ").append(str));
  }

  void too_few(const char *prog, std::string op) {
    print_usage(prog);
    throw std::invalid_argument(std::move(op.append(": too few arguments")));
  }

#if defined(USE_ITERATIONS)
  void print_usage(const char *prog) {
    std::cerr << "Usage:\n"
              << "\t" << prog
              << " {dgesv,sgesv,dgetrs,sgetrs} <n> <nrhs> <iters>\n"
              << "\t" << prog << " {dgels,sgels} <m> <n> <nrhs> <iters>\n"
              << "\t" << prog << " {dgetri,sgetri} <n> <iters>\n"
              << "\t" << prog << " {dtptri,stptri,dtrtri,strtri} <n> <iters>\n"
              << "\t" << prog << " {dpotrf,spotrf} <n> <iters>\n"
              << "\t" << prog << " {dgetrf,sgetrf} <m> <n> <iters>\n";
  }

  void get_iterations(int argc, const char *const *argv, int idx) {
    if (argc > idx)
      util::to_scalar(argv[idx], g_iters);
    assert_positive(g_iters, "iters");
  }
#else  // !defined(USE_ITERATIONS)
  void print_usage(const char *prog) {
    std::cerr << "Usage:\n"
              << "\t" << prog << " {dgesv,sgesv,dgetrs,sgetrs} <n> <nrhs>\n"
              << "\t" << prog << " {dgels,sgels} <m> <n> <nrhs>\n"
              << "\t" << prog << " {dgetri,sgetri} <n>\n"
              << "\t" << prog << " {dtptri,stptri,dtrtri,strtri} <n>\n"
              << "\t" << prog << " {dpotrf,spotrf} <n>\n"
              << "\t" << prog << " {dgetrf,sgetrf} <m> <n>\n";
  }

  void get_iterations(int, const char *const *, int) {}
#endif // defined(USE_ITERATIONS)
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
