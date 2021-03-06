#include <timeprinter/printer.hpp>
#include <util/cuda_utils.hpp>
#include <util/to_scalar.hpp>
#include <util/unique_handle.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cassert>
#include <random>

#define NO_INLINE __attribute__((noinline))

#if CUDART_VERSION < 11000
#define NO_IPA __attribute__((noipa))
#endif // CUDART_VERSION < 11000

// disable semantic equivalency optimization when not doing computation
#if defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)
#define NO_CLONE __attribute__((noclone, no_icf))
#else // !defined(USE_ITERATIONS)
#define NO_CLONE __attribute__(())
#endif // defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)

namespace {
tp::printer g_tpr;

#if defined(USE_ITERATIONS)
std::size_t g_iters;
#endif

struct compute_params {
  std::size_t M = 0;
  std::size_t N = 0;
  std::size_t Nrhs = 0;
};

std::ostream &operator<<(std::ostream &os, const compute_params &p) {
  if (p.M)
    os << "M=" << p.M << " ";
  if (p.N)
    os << "N=" << p.N << " ";
  if (p.Nrhs)
    os << "Nrhs=" << p.Nrhs << " ";
  return os;
}

template <auto Func>
struct func_obj : std::integral_constant<decltype(Func), Func> {};

cusolverDnHandle_t cusolverdn_create() {
  cusolverDnHandle_t handle;
  auto status = cusolverDnCreate(&handle);
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error("Error creating cuSOLVER");
  return handle;
}

void cusolverdn_destroy(cusolverDnHandle_t handle) {
  auto status = cusolverDnDestroy(handle);
  if (status != CUSOLVER_STATUS_SUCCESS)
    std::cerr << "Error destroying cuSOLVER";
}

using cusolverdn_handle =
    util::unique_handle<cusolverDnHandle_t, func_obj<cusolverdn_destroy>>;

namespace detail {
#if CUDART_VERSION < 11010
using index_type = cusolver_int_t;
#else
using index_type = std::int64_t;
#endif // CUDART_VERSION < 11010

#if defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)
void handle_error(int) {}
void cusolver_error(const char *, cusolverStatus_t) {}
#else  // !defined(USE_ITERATIONS)
void handle_error(int info) {
  if (info > 0)
    throw std::runtime_error(
        std::string("The solution could not be computed; info=")
            .append(std::to_string(info)));
  if (info < 0)
    throw std::runtime_error(
        std::string("Invalid parameter; info=").append(std::to_string(info)));
}

void cusolver_error(const char *func, cusolverStatus_t status) {
  throw std::runtime_error(std::string(func)
                               .append(" error, status = ")
                               .append(std::to_string(status)));
}
#endif // defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)

template <typename Real> struct any_call {
  struct impl {
    template <typename... Args>
    constexpr cusolverStatus_t operator()(Args &&...) const {
      return CUSOLVER_STATUS_SUCCESS;
    }
  };
  using type = Real;
  static constexpr auto query = impl{};
  static constexpr auto query_str = "";
  static constexpr auto compute = impl{};
  static constexpr auto compute_str = "";
};

#if defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)
template <typename Real> using gesv_call = any_call<Real>;
template <typename Real> using gels_call = any_call<Real>;
template <typename Real> using getrf_call = any_call<Real>;
template <typename Real> using getrs_call = any_call<Real>;
template <typename Real> using trtri_call = any_call<Real>;
template <typename Real> using potrf_call = any_call<Real>;
#else // !defined(USE_ITERATIONS)
#define DEFINE_CALL_MEMBERS(prefix, prec)                                      \
  static constexpr auto query = cusolverDn##prec##prefix##_bufferSize;         \
  static constexpr char query_str[] =                                          \
      "cusolverDn" #prec #prefix "_bufferSize";                                \
  static constexpr auto compute = cusolverDn##prec##prefix;                    \
  static constexpr char compute_str[] = "cusolverDn" #prec #prefix

#define DEFINE_CALL_ANY(prefix, prec_single, prec_double)                      \
  template <typename> struct prefix##_call {};                                 \
  template <> struct prefix##_call<float> {                                    \
    DEFINE_CALL_MEMBERS(prefix, prec_single);                                  \
  };                                                                           \
  template <> struct prefix##_call<double> {                                   \
    DEFINE_CALL_MEMBERS(prefix, prec_double);                                  \
  }

#define DEFINE_CALL_IRS(prefix) DEFINE_CALL_ANY(prefix, SS, DD)
#define DEFINE_CALL(prefix) DEFINE_CALL_ANY(prefix, S, D)

#if CUDART_VERSION >= 10020
DEFINE_CALL_IRS(gesv);
#else  // CUDART_VERSION < 10020
template <typename Real> using gesv_call = any_call<Real>;
#endif // CUDART_VERSION >= 10020

#if CUDART_VERSION >= 11000
DEFINE_CALL_IRS(gels);
#else  // CUDART_VERSION < 11000
template <typename Real> using gels_call = any_call<Real>;
#endif // CUDART_VERSION >= 11000

#if CUDART_VERSION < 11040
DEFINE_CALL(trtri);
#else  // CUDART_VERSION >= 11040
template <typename T> struct trtri_call {
  static constexpr auto compute = cusolverDnXtrtri;
  static constexpr auto compute_str = "cusolverDnXtrtri";
  static constexpr auto query = cusolverDnXtrtri_bufferSize;
  static constexpr auto query_str = "cusolverDnXtrtri_bufferSize";
};
#endif // CUDART_VERSION < 11040

#if CUDART_VERSION < 11010
DEFINE_CALL(getrf);
template <typename> struct getrs_call {};
template <> struct getrs_call<float> {
  static constexpr auto compute = cusolverDnSgetrs;
  static constexpr char compute_str[] = "cusolverDnSgetrs";
};
template <> struct getrs_call<double> {
  static constexpr auto compute = cusolverDnDgetrs;
  static constexpr char compute_str[] = "cusolverDnDgetrs";
};
#else  // CUDART_VERSION >= 11010
template <typename T> struct getrf_call {
  static constexpr auto compute = cusolverDnXgetrf;
  static constexpr auto compute_str = "cusolverDnXgetrf";
  static constexpr auto query = cusolverDnXgetrf_bufferSize;
  static constexpr auto query_str = "cusolverDnXgetrf_bufferSize";
};
template <typename T> struct getrs_call {
  static constexpr auto compute = cusolverDnXgetrs;
  static constexpr auto compute_str = "cusolverDnXgetrs";
};
#endif // CUDART_VERSION < 11010

#if CUDART_VERSION < 11000
DEFINE_CALL(potrf);
#elif CUDART_VERSION >= 11010
template <typename T> struct potrf_call {
  static constexpr auto compute = cusolverDnXpotrf;
  static constexpr auto compute_str = "cusolverDnXpotrf";
  static constexpr auto query = cusolverDnXpotrf_bufferSize;
  static constexpr auto query_str = "cusolverDnXpotrf_bufferSize";
};
#else  // CUDART_VERSION >= 11000
template <typename T> struct potrf_call {
  static constexpr auto compute = cusolverDnPotrf;
  static constexpr auto compute_str = "cusolverDnPotrf";
  static constexpr auto query = cusolverDnPotrf_bufferSize;
  static constexpr auto query_str = "cusolverDnPotrf_bufferSize";
};
#endif // CUDART_VERSION < 11000
#endif // defined(USE_ITERATIONS) && !defined(DO_COMPUTATION)

template <typename> struct cuda_data_type {};

template <>
struct cuda_data_type<float>
    : std::integral_constant<decltype(CUDA_R_32F), CUDA_R_32F> {};

template <>
struct cuda_data_type<double>
    : std::integral_constant<decltype(CUDA_R_64F), CUDA_R_64F> {};

#if CUDART_VERSION < 11040
std::pair<int, int> cudart_separate_version(int version) {
  return {version / 1000, (version % 1000) / 10};
}

std::string unsupported_version(std::string_view feature,
                                std::string_view required) {
  auto [major, minor] = cudart_separate_version(CUDART_VERSION);
  return std::string(feature)
      .append(" requires CUDA Toolkit v")
      .append(required)
      .append(" or higher, found CUDA runtime v")
      .append(std::to_string(major))
      .append(".")
      .append(std::to_string(minor));
}
#endif // CUDART_VERSION < 11040

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

  util::buffer<Real> a(N * N);
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
template <typename T>
std::pair<util::device_buffer<T>, util::buffer<T>>
work_buffers(std::size_t workspace_device, std::size_t workspace_host) {
  return {util::device_buffer<std::uint8_t>{workspace_device},
          workspace_host ? util::buffer<std::uint8_t>{workspace_host}
                         : util::buffer<std::uint8_t>{}};
}

template <typename Caller, typename... Args>
void query(cusolverdn_handle &h, Args... args) {
  auto status = Caller::query(h, args...);
  if (status != CUSOLVER_STATUS_SUCCESS)
    cusolver_error(Caller::query_str, status);
}

template <typename Caller, typename... Args>
void compute(cusolverdn_handle &h,
             util::device_buffer<cusolver_int_t> &dev_info, Args... args) {
  util::zero(dev_info);
  auto status = Caller::compute(h, args..., dev_info.get());
  if (status != CUSOLVER_STATUS_SUCCESS)
    cusolver_error(Caller::compute_str, status);
  cusolver_int_t info;
  util::copy(dev_info, 1, &info);
  handle_error(info);
}

#if defined(USE_ITERATIONS)
#if CUDART_VERSION < 11040
template <typename Real>
void trtri(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
  int lwork = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> a_copy{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{a, a_copy.size()}, a_copy);
    query<trtri_call<Real>>(handle, uplo, diag, N, a_copy.get(), N, &lwork);
    util::device_buffer<Real> dev_work{static_cast<std::size_t>(lwork)};
    compute<trtri_call<Real>>(handle, dev_info, uplo, diag, N, a_copy.get(), N,
                              dev_work.get(), lwork);
  }
}
#else  // CUDART_VERSION >= 11040
template <typename Real>
void trtri(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
  std::size_t workspace_device = 0;
  std::size_t workspace_host = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> a_copy{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{a, a_copy.size()}, a_copy);
    query<trtri_call<Real>>(handle, uplo, diag, N, cuda_data_type<Real>::value,
                            a_copy.get(), N, &workspace_device,
                            &workspace_host);
    auto [dev_work, host_work] =
        work_buffers<std::uint8_t>(workspace_device, workspace_host);
    compute<trtri_call<Real>>(handle, dev_info, uplo, diag, N,
                              cuda_data_type<Real>::value, a_copy.get(), N,
                              dev_work.get(), workspace_device, host_work.get(),
                              workspace_host);
  }
}
#endif // CUDART_VERSION < 11040

#if CUDART_VERSION < 11010
template <typename Real>
void getrf(cusolverdn_handle &handle, std::size_t M, std::size_t N, Real *a,
           index_type *ipiv) {
  int lwork = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> a_copy{M * N};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{a, a_copy.size()}, a_copy);
    query<getrf_call<Real>>(handle, M, N, a_copy.get(), M, &lwork);
    util::device_buffer<Real> dev_work{static_cast<std::size_t>(lwork)};
    compute<getrf_call<Real>>(handle, dev_info, M, N, a_copy.get(), M,
                              dev_work.get(), ipiv);
  }
}
#else  // CUDART_VERSION >= 11010
template <typename Real>
void getrf(cusolverdn_handle &handle, std::size_t M, std::size_t N, Real *a,
           index_type *ipiv) {
  std::size_t workspace_device = 0;
  std::size_t workspace_host = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> a_copy{M * N};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{a, a_copy.size()}, a_copy);
    query<getrf_call<Real>>(handle, nullptr, M, N, cuda_data_type<Real>::value,
                            a_copy.get(), M, cuda_data_type<Real>::value,
                            &workspace_device, &workspace_host);
    auto [dev_work, host_work] =
        work_buffers<std::uint8_t>(workspace_device, workspace_host);
    compute<getrf_call<Real>>(
        handle, dev_info, nullptr, M, N, cuda_data_type<Real>::value,
        a_copy.get(), M, ipiv, cuda_data_type<Real>::value, dev_work.get(),
        workspace_device, host_work.get(), workspace_host);
  }
}
#endif // CUDART_VERSION < 11010

#if CUDART_VERSION < 11010
template <typename Real>
void getrs(cusolverdn_handle &handle, std::size_t N, std::size_t Nrhs,
           const Real *a, const index_type *ipiv, Real *b) {
  constexpr cublasOperation_t op = CUBLAS_OP_N;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> b_copy{N * Nrhs};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{b, b_copy.size()}, b_copy);
    compute<getrs_call<Real>>(handle, dev_info, op, N, Nrhs, a, N, ipiv,
                              b_copy.get(), N);
  }
}
#else  // CUDART_VERSION >= 11010
template <typename Real>
void getrs(cusolverdn_handle &handle, std::size_t N, std::size_t Nrhs,
           const Real *a, const index_type *ipiv, Real *b) {
  constexpr cublasOperation_t op = CUBLAS_OP_N;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> b_copy{N * Nrhs};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{b, b_copy.size()}, b_copy);
    compute<getrs_call<Real>>(handle, dev_info, nullptr, op, N, Nrhs,
                              cuda_data_type<Real>::value, a, N, ipiv,
                              cuda_data_type<Real>::value, b_copy.get(), N);
  }
}
#endif // CUDART_VERSION < 11010

#if CUDART_VERSION < 11000
// cusolverDn<t>potrf()
template <typename Real>
void potrf(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  int lwork = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> a_copy{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{a, a_copy.size()}, a_copy);
    query<potrf_call<Real>>(handle, uplo, N, a_copy.get(), N, &lwork);
    util::device_buffer<Real> dev_work{static_cast<std::size_t>(lwork)};
    compute<potrf_call<Real>>(handle, dev_info, uplo, N, a_copy.get(), N,
                              dev_work.get(), lwork);
  }
}
#elif CUDART_VERSION >= 11010
template <typename Real>
void potrf(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  std::size_t workspace_device = 0;
  std::size_t workspace_host = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> a_copy{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{a, a_copy.size()}, a_copy);
    query<potrf_call<Real>>(
        handle, nullptr, uplo, N, cuda_data_type<Real>::value, a_copy.get(), N,
        cuda_data_type<Real>::value, &workspace_device, &workspace_host);
    auto [dev_work, host_work] =
        work_buffers<std::uint8_t>(workspace_device, workspace_host);
    compute<potrf_call<Real>>(
        handle, dev_info, nullptr, uplo, N, cuda_data_type<Real>::value,
        a_copy.get(), N, cuda_data_type<Real>::value, dev_work.get(),
        workspace_device, host_work.get(), workspace_host);
  }
}
#else  // CUDART_VERSION >= 11000
// cusolverDnPotrf()
template <typename Real>
void potrf(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  std::size_t workspace = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> a_copy{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{a, a_copy.size()}, a_copy);
    query<potrf_call<Real>>(handle, nullptr, uplo, N,
                            cuda_data_type<Real>::value, a_copy.get(), N,
                            cuda_data_type<Real>::value, &workspace);
    util::device_buffer<std::uint8_t> dev_work{workspace};
    compute<potrf_call<Real>>(handle, dev_info, nullptr, uplo, N,
                              cuda_data_type<Real>::value, a_copy.get(), N,
                              cuda_data_type<Real>::value, dev_work.get(),
                              workspace);
  }
}
#endif // CUDART_VERSION < 11000

template <typename Real>
void gesv(cusolverdn_handle &handle, std::size_t N, std::size_t Nrhs, Real *a,
          Real *b, Real *x, cusolver_int_t *ipiv) {
  std::size_t work_bytes = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> a_copy{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{a, a_copy.size()}, a_copy);
    query<gesv_call<Real>>(handle, N, Nrhs, nullptr, N, nullptr, nullptr, N,
                           nullptr, N, nullptr, &work_bytes);
    cusolver_int_t iters = 0;
    util::device_buffer<std::uint8_t> dev_work{work_bytes};
    compute<gesv_call<Real>>(handle, dev_info, N, Nrhs, a_copy.get(), N, ipiv,
                             b, N, x, N, dev_work.get(), work_bytes, &iters);
  }
}

template <typename Real>
void gels(cusolverdn_handle &handle, std::size_t M, std::size_t N,
          std::size_t Nrhs, Real *a, Real *b, Real *x) {
  std::size_t work_bytes = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> a_copy{N * N};
  for (size_t i = 0; i < g_iters; i++) {
    util::copy(util::device_buffer_view{a, a_copy.size()}, a_copy);
    query<gels_call<Real>>(handle, M, N, Nrhs, nullptr, M, nullptr, M, nullptr,
                           N, nullptr, &work_bytes);
    cusolver_int_t iters = 0;
    util::device_buffer<std::uint8_t> dev_work{work_bytes};
    compute<gels_call<Real>>(handle, dev_info, M, N, Nrhs, a_copy.get(), M, b,
                             M, x, N, dev_work.get(), work_bytes, &iters);
  }
}
#else // !defined(USE_ITERATIONS)
#if CUDART_VERSION < 11040
template <typename Real>
void trtri(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
  int lwork = 0;
  query<trtri_call<Real>>(handle, uplo, diag, N, a, N, &lwork);
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> dev_work{static_cast<std::size_t>(lwork)};
  compute<trtri_call<Real>>(handle, dev_info, uplo, diag, N, a, N,
                            dev_work.get(), lwork);
}
#else  // CUDART_VERSION >= 11040
template <typename Real>
void trtri(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
  std::size_t workspace_device = 0;
  std::size_t workspace_host = 0;
  query<trtri_call<Real>>(handle, uplo, diag, N, cuda_data_type<Real>::value, a,
                          N, &workspace_device, &workspace_host);
  auto [dev_work, host_work] =
      work_buffers<std::uint8_t>(workspace_device, workspace_host);
  util::device_buffer<cusolver_int_t> dev_info{1};
  compute<trtri_call<Real>>(handle, dev_info, uplo, diag, N,
                            cuda_data_type<Real>::value, a, N, dev_work.get(),
                            workspace_device, host_work.get(), workspace_host);
}
#endif // CUDART_VERSION < 11040

#if CUDART_VERSION < 11010
template <typename Real>
void getrf(cusolverdn_handle &handle, std::size_t M, std::size_t N, Real *a,
           index_type *ipiv) {
  int lwork = 0;
  query<getrf_call<Real>>(handle, M, N, a, M, &lwork);
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> dev_work{static_cast<std::size_t>(lwork)};
  compute<getrf_call<Real>>(handle, dev_info, M, N, a, M, dev_work.get(), ipiv);
}
#else  // CUDART_VERSION >= 11010
template <typename Real>
void getrf(cusolverdn_handle &handle, std::size_t M, std::size_t N, Real *a,
           index_type *ipiv) {
  std::size_t workspace_device = 0;
  std::size_t workspace_host = 0;
  query<getrf_call<Real>>(handle, nullptr, M, N, cuda_data_type<Real>::value, a,
                          M, cuda_data_type<Real>::value, &workspace_device,
                          &workspace_host);
  auto [dev_work, host_work] =
      work_buffers<std::uint8_t>(workspace_device, workspace_host);
  util::device_buffer<cusolver_int_t> dev_info{1};
  compute<getrf_call<Real>>(handle, dev_info, nullptr, M, N,
                            cuda_data_type<Real>::value, a, M, ipiv,
                            cuda_data_type<Real>::value, dev_work.get(),
                            workspace_device, host_work.get(), workspace_host);
}
#endif // CUDART_VERSION < 11010

#if CUDART_VERSION < 11010
template <typename Real>
void getrs(cusolverdn_handle &handle, std::size_t N, std::size_t Nrhs,
           const Real *a, const index_type *ipiv, Real *b) {
  constexpr cublasOperation_t op = CUBLAS_OP_N;
  util::device_buffer<cusolver_int_t> dev_info{1};
  compute<getrs_call<Real>>(handle, dev_info, op, N, Nrhs, a, N, ipiv, b, N);
}
#else  // CUDART_VERSION >= 11010
template <typename Real>
void getrs(cusolverdn_handle &handle, std::size_t N, std::size_t Nrhs,
           const Real *a, const index_type *ipiv, Real *b) {
  constexpr cublasOperation_t op = CUBLAS_OP_N;
  util::device_buffer<cusolver_int_t> dev_info{1};
  compute<getrs_call<Real>>(handle, dev_info, nullptr, op, N, Nrhs,
                            cuda_data_type<Real>::value, a, N, ipiv,
                            cuda_data_type<Real>::value, b, N);
}
#endif // CUDART_VERSION < 11010

#if CUDART_VERSION < 11000
// cusolverDn<t>potrf()
template <typename Real>
void potrf(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  int lwork = 0;
  query<potrf_call<Real>>(handle, uplo, N, a, N, &lwork);
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<Real> dev_work{static_cast<std::size_t>(lwork)};
  compute<potrf_call<Real>>(handle, dev_info, uplo, N, a, N, dev_work.get(),
                            lwork);
}
#elif CUDART_VERSION >= 11010
template <typename Real>
void potrf(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  std::size_t workspace_device = 0;
  std::size_t workspace_host = 0;
  query<potrf_call<Real>>(handle, nullptr, uplo, N, cuda_data_type<Real>::value,
                          a, N, cuda_data_type<Real>::value, &workspace_device,
                          &workspace_host);
  auto [dev_work, host_work] =
      work_buffers<std::uint8_t>(workspace_device, workspace_host);
  util::device_buffer<cusolver_int_t> dev_info{1};
  compute<potrf_call<Real>>(handle, dev_info, nullptr, uplo, N,
                            cuda_data_type<Real>::value, a, N,
                            cuda_data_type<Real>::value, dev_work.get(),
                            workspace_device, host_work.get(), workspace_host);
}
#else  // CUDART_VERSION >= 11000
// cusolverDnPotrf()
template <typename Real>
void potrf(cusolverdn_handle &handle, std::size_t N, Real *a) {
  constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  std::size_t workspace = 0;
  query<potrf_call<Real>>(handle, nullptr, uplo, N, cuda_data_type<Real>::value,
                          a, N, cuda_data_type<Real>::value, &workspace);
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<std::uint8_t> dev_work{workspace};
  compute<potrf_call<Real>>(
      handle, dev_info, nullptr, uplo, N, cuda_data_type<Real>::value, a, N,
      cuda_data_type<Real>::value, dev_work.get(), workspace);
}
#endif // CUDART_VERSION < 11000

template <typename Real>
void gesv(cusolverdn_handle &handle, std::size_t N, std::size_t Nrhs, Real *a,
          Real *b, Real *x, cusolver_int_t *ipiv) {
  std::size_t work_bytes = 0;
  query<gesv_call<Real>>(handle, N, Nrhs, nullptr, N, nullptr, nullptr, N,
                         nullptr, N, nullptr, &work_bytes);
  cusolver_int_t iters = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<std::uint8_t> dev_work{work_bytes};
  compute<gesv_call<Real>>(handle, dev_info, N, Nrhs, a, N, ipiv, b, N, x, N,
                           dev_work.get(), work_bytes, &iters);
  std::cerr << "iterations = " << iters << "\n";
}

template <typename Real>
void gels(cusolverdn_handle &handle, std::size_t M, std::size_t N,
          std::size_t Nrhs, Real *a, Real *b, Real *x) {
  std::size_t work_bytes = 0;
  query<gels_call<Real>>(handle, M, N, Nrhs, nullptr, M, nullptr, M, nullptr, N,
                         nullptr, &work_bytes);
  cusolver_int_t iters = 0;
  util::device_buffer<cusolver_int_t> dev_info{1};
  util::device_buffer<std::uint8_t> dev_work{work_bytes};
  compute<gels_call<Real>>(handle, dev_info, M, N, Nrhs, a, M, b, M, x, N,
                           dev_work.get(), work_bytes, &iters);
  std::cerr << "iterations = " << iters << "\n";
}
#endif // defined(USE_ITERATIONS)
} // namespace impl

template <typename Real>
NO_INLINE NO_CLONE void trtri(cusolverdn_handle &handle, std::size_t N,
                              Real *a) {
  tp::sampler smp(g_tpr);
  impl::trtri(handle, N, a);
}

template <typename Real>
NO_INLINE NO_CLONE void getrf(cusolverdn_handle &handle, std::size_t M,
                              std::size_t N, Real *a, index_type *ipiv) {
  tp::sampler smp(g_tpr);
  impl::getrf(handle, M, N, a, ipiv);
}

template <typename Real>
NO_INLINE NO_CLONE void getrs(cusolverdn_handle &handle, std::size_t N,
                              std::size_t Nrhs, const Real *a,
                              const index_type *ipiv, Real *b) {
  tp::sampler smp(g_tpr);
  impl::getrs(handle, N, Nrhs, a, ipiv, b);
}

template <typename Real>
NO_INLINE NO_CLONE void potrf(cusolverdn_handle &handle, std::size_t N,
                              Real *a) {
  tp::sampler smp(g_tpr);
  impl::potrf(handle, N, a);
}

#if CUDART_VERSION < 10020
template <typename Real>
NO_IPA void gesv(cusolverdn_handle &, std::size_t, std::size_t, Real *, Real *,
                 Real *, cusolver_int_t *) {
  throw std::runtime_error(
      unsupported_version("cusolverDn<t1><t2>gesv()", "10.2"));
}
#else  // CUDART_VERSION >= 10020
template <typename Real>
NO_INLINE NO_CLONE void gesv(cusolverdn_handle &handle, std::size_t N,
                             std::size_t Nrhs, Real *a, Real *b, Real *x,
                             cusolver_int_t *ipiv) {
  tp::sampler smp(g_tpr);
  impl::gesv(handle, N, Nrhs, a, b, x, ipiv);
}
#endif // CUDART_VERSION < 10020

#if CUDART_VERSION < 11000
template <typename Real>
NO_IPA void gels(cusolverdn_handle &, std::size_t, std::size_t, std::size_t,
                 Real *, Real *, Real *) {
  throw std::runtime_error(
      unsupported_version("cusolverDn<t1><t2>gels()", "11.0"));
}
#else  // CUDART_VERSION >= 11000
template <typename Real>
NO_INLINE NO_CLONE void gels(cusolverdn_handle &handle, std::size_t M,
                             std::size_t N, std::size_t Nrhs, Real *a, Real *b,
                             Real *x) {
  tp::sampler smp(g_tpr);
  impl::gels(handle, M, N, Nrhs, a, b, x);
}
#endif // CUDART_VERSION < 11000
} // namespace compute

template <typename Real>
void trtri_impl(cusolverdn_handle &handle, std::size_t N,
                std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{1.0, 2.0};
  util::buffer<Real> a{N * N};
  std::fill(a.begin(), a.end(), Real{});
  fill_upper_triangular(a.begin(), a.end(), N, [&]() { return dist(engine); });
  smp.do_sample();
  util::device_buffer dev_a{a.begin(), a.end()};
  compute::trtri(handle, N, dev_a.get());
  util::copy(dev_a, dev_a.size(), a.begin());
}

template <typename Real>
void getrf_impl(cusolverdn_handle &handle, std::size_t M, std::size_t N,
                std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  util::buffer<Real> a{M * N};
  util::buffer<index_type> ipiv{std::min(M, N)};
  std::generate(a.begin(), a.end(), [&]() { return dist(engine); });

  smp.do_sample();

  util::device_buffer dev_a{a.begin(), a.end()};
  util::device_buffer<index_type> dev_ipiv{ipiv.size()};
  compute::getrf(handle, M, N, dev_a.get(), dev_ipiv.get());
  util::copy(dev_a, dev_a.size(), a.begin());
  util::copy(dev_ipiv, dev_ipiv.size(), ipiv.begin());
}

template <typename Real>
void potrf_impl(cusolverdn_handle &handle, std::size_t N,
                std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  util::buffer<Real> a = upper_dd_matrix<Real>(N, engine);
  util::device_buffer dev_a{a.begin(), a.end()};
  compute::potrf(handle, N, dev_a.get());
  util::copy(dev_a, dev_a.size(), a.begin());
}

template <typename Real>
void getrs_impl(cusolverdn_handle &handle, std::size_t N, std::size_t Nrhs,
                std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  util::buffer<Real> a{N * N};
  util::buffer<Real> b{N * Nrhs};
  util::buffer<index_type> ipiv{N};
  auto gen = [&]() { return dist(engine); };
  std::generate(a.begin(), a.end(), gen);
  std::generate(b.begin(), b.end(), gen);

  smp.do_sample();

  util::device_buffer dev_a{a.begin(), a.end()};
  util::device_buffer dev_b{b.begin(), b.end()};
  util::device_buffer<index_type> dev_ipiv{ipiv.size()};
  compute::getrf(handle, N, N, dev_a.get(), dev_ipiv.get());
  compute::getrs(handle, N, Nrhs, dev_a.get(), dev_ipiv.get(), dev_b.get());
  util::copy(dev_b, dev_b.size(), b.begin());
}

#if CUDART_VERSION < 10020
template <typename Real>
void gesv_impl(cusolverdn_handle &handle, std::size_t, std::size_t,
               std::mt19937_64 &) {
  compute::gesv<Real>(handle, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
#else  // CUDART_VERSION >= 10020
template <typename Real>
void gesv_impl(cusolverdn_handle &handle, std::size_t N, std::size_t Nrhs,
               std::mt19937_64 &engine) {
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  util::buffer<Real> a{N * N};
  util::buffer<Real> b{N * Nrhs};
  util::buffer<Real> x{N * Nrhs};
  util::buffer<cusolver_int_t> ipiv{N};
  auto gen = [&]() { return dist(engine); };
  std::generate(a.begin(), a.end(), gen);
  std::generate(b.begin(), b.end(), gen);

  smp.do_sample();

  util::device_buffer dev_a{a.begin(), a.end()};
  util::device_buffer dev_b{b.begin(), b.end()};
  util::device_buffer<Real> dev_x{x.size()};
  util::device_buffer<cusolver_int_t> dev_ipiv{ipiv.size()};
  compute::gesv(handle, N, Nrhs, dev_a.get(), dev_b.get(), dev_x.get(),
                ipiv.get());
  util::copy(dev_x, dev_x.size(), x.begin());
  util::copy(dev_ipiv, dev_ipiv.size(), ipiv.begin());
  util::copy(dev_a, dev_a.size(), a.begin());
}
#endif // CUDART_VERSION < 10020

#if CUDART_VERSION < 11000
template <typename Real>
void gels_impl(cusolverdn_handle &handle, std::size_t, std::size_t, std::size_t,
               std::mt19937_64 &) {
  compute::gels<Real>(handle, 0, 0, 0, nullptr, nullptr, nullptr);
}
#else  // CUDART_VERSION >= 11000
template <typename Real>
void gels_impl(cusolverdn_handle &handle, std::size_t M, std::size_t N,
               std::size_t Nrhs, std::mt19937_64 &engine) {
  assert(N <= M);
  tp::sampler smp(g_tpr);
  std::uniform_real_distribution<Real> dist{0.0, 1.0};
  util::buffer<Real> a{M * N};
  util::buffer<Real> b{M * Nrhs};
  util::buffer<Real> x{N * Nrhs};
  auto gen = [&]() { return dist(engine); };
  std::generate(a.begin(), a.end(), gen);
  std::generate(b.begin(), b.end(), gen);

  smp.do_sample();

  util::device_buffer dev_a{a.begin(), a.end()};
  util::device_buffer dev_b{b.begin(), b.end()};
  util::device_buffer<Real> dev_x{x.size()};
  compute::gels(handle, M, N, Nrhs, dev_a.get(), dev_b.get(), dev_x.get());
  util::copy(dev_x, dev_x.size(), x.begin());
  util::copy(dev_a, dev_a.size(), a.begin());
}
#endif // CUDART_VERSION < 11000
} // namespace detail

NO_INLINE void dtrtri(cusolverdn_handle &handle, compute_params p,
                      std::mt19937_64 &engine) {
  detail::trtri_impl<double>(handle, p.N, engine);
}

NO_INLINE void strtri(cusolverdn_handle &handle, compute_params p,
                      std::mt19937_64 &engine) {
  detail::trtri_impl<float>(handle, p.N, engine);
}

NO_INLINE void dgetrf(cusolverdn_handle &handle, compute_params p,
                      std::mt19937_64 &engine) {
  detail::getrf_impl<double>(handle, p.M, p.N, engine);
}

NO_INLINE void sgetrf(cusolverdn_handle &handle, compute_params p,
                      std::mt19937_64 &engine) {
  detail::getrf_impl<float>(handle, p.M, p.N, engine);
}

NO_INLINE void dgetrs(cusolverdn_handle &handle, compute_params p,
                      std::mt19937_64 &engine) {
  detail::getrs_impl<double>(handle, p.N, p.Nrhs, engine);
}

NO_INLINE void sgetrs(cusolverdn_handle &handle, compute_params p,
                      std::mt19937_64 &engine) {
  detail::getrs_impl<float>(handle, p.N, p.Nrhs, engine);
}

NO_INLINE void dgesv(cusolverdn_handle &handle, compute_params p,
                     std::mt19937_64 &engine) {
  detail::gesv_impl<double>(handle, p.N, p.Nrhs, engine);
}

NO_INLINE void sgesv(cusolverdn_handle &handle, compute_params p,
                     std::mt19937_64 &engine) {
  detail::gesv_impl<float>(handle, p.N, p.Nrhs, engine);
}

NO_INLINE void dgels(cusolverdn_handle &handle, compute_params p,
                     std::mt19937_64 &engine) {
  detail::gels_impl<double>(handle, p.M, p.N, p.Nrhs, engine);
}

NO_INLINE void sgels(cusolverdn_handle &handle, compute_params p,
                     std::mt19937_64 &engine) {
  detail::gels_impl<float>(handle, p.M, p.N, p.Nrhs, engine);
}

NO_INLINE void dpotrf(cusolverdn_handle &handle, compute_params p,
                      std::mt19937_64 &engine) {
  detail::potrf_impl<double>(handle, p.N, engine);
}

NO_INLINE void spotrf(cusolverdn_handle &handle, compute_params p,
                      std::mt19937_64 &engine) {
  detail::potrf_impl<float>(handle, p.N, engine);
}

struct cmdargs {
  using work_func = decltype(&dtrtri);
  work_func func = nullptr;
  compute_params params = {};

  cmdargs(int argc, const char *const *argv) {
    const char *prog = argv[0];
    if (argc < 2) {
      usage(prog);
      throw std::invalid_argument("Not enough arguments");
    }
    std::string op_type(argv[1]);
    std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    func = get_work_type(op_type);
    if (func == dgetrf || func == sgetrf) {
      if (argc < 4)
        throw_too_few(prog, op_type);
      util::to_scalar(argv[2], params.M);
      check_positive(params.M, "m");
      util::to_scalar(argv[3], params.N);
      check_positive(params.N, "n");
      get_iterations(argc, argv, 4);
    } else if (func == dgetrs || func == sgetrs || func == dgesv ||
               func == sgesv) {
      if (argc < 4)
        throw_too_few(prog, op_type);
      util::to_scalar(argv[2], params.N);
      check_positive(params.N, "n");
      util::to_scalar(argv[3], params.Nrhs);
      check_positive(params.Nrhs, "n_rhs");
      get_iterations(argc, argv, 4);
    } else if (func == dgels || func == sgels) {
      if (argc < 5)
        throw_too_few(prog, op_type);
      util::to_scalar(argv[2], params.M);
      check_positive(params.M, "m");
      util::to_scalar(argv[3], params.N);
      check_positive(params.N, "n");
      util::to_scalar(argv[4], params.Nrhs);
      check_positive(params.Nrhs, "n_rhs");
      if (params.N > params.M)
        throw std::invalid_argument("n must not be greater than m");
      get_iterations(argc, argv, 5);
    } else if (func == dtrtri || func == strtri || func == dpotrf ||
               func == spotrf) {
      if (argc < 3)
        throw_too_few(prog, op_type);
      util::to_scalar(argv[2], params.N);
      check_positive(params.N, "n");
      get_iterations(argc, argv, 3);
    }
  }

private:
  static void check_positive(std::size_t val, const char *val_str) {
    if (!val)
      throw std::invalid_argument(
          std::string(val_str).append(" must be a positive integer"));
    assert(val);
  }

  static void throw_too_few(const char *prog, std::string &op_type) {
    usage(prog);
    throw std::invalid_argument(op_type.append(": too few arguments"));
  }

  static work_func get_work_type(const std::string &op_type) {
#define WORK_RETURN_IF(op_type, arg)                                           \
  if (op_type == #arg)                                                         \
  return arg

    WORK_RETURN_IF(op_type, dtrtri);
    WORK_RETURN_IF(op_type, strtri);
    WORK_RETURN_IF(op_type, dgetrf);
    WORK_RETURN_IF(op_type, sgetrf);
    WORK_RETURN_IF(op_type, dgetrs);
    WORK_RETURN_IF(op_type, sgetrs);
    WORK_RETURN_IF(op_type, dgesv);
    WORK_RETURN_IF(op_type, sgesv);
    WORK_RETURN_IF(op_type, dgels);
    WORK_RETURN_IF(op_type, sgels);
    WORK_RETURN_IF(op_type, dpotrf);
    WORK_RETURN_IF(op_type, spotrf);
    throw std::invalid_argument(
        std::string("Unknown work type ").append(op_type));
  }

#if defined(USE_ITERATIONS)
  static void usage(const char *prog) {
    std::cerr << "Usage:\n"
              << "\t" << prog << " {dtrtri,strtri} <n> <iters>\n"
              << "\t" << prog << " {dpotrf,spotrf} <n> <iters>\n"
              << "\t" << prog << " {dgetrf,sgetrf} <m> <n> <iters>\n"
              << "\t" << prog
              << " {dgetrs,sgetrs,dgesv,sgesv} <n> <n_rhs> <iters>\n"
              << "\t" << prog << " {dgels,sgels} <n> <m> <n_rhs> <iters>\n";
  }

  static void get_iterations(int argc, const char *const *argv, int idx) {
    if (argc > idx)
      util::to_scalar(argv[idx], g_iters);
    check_positive(g_iters, "iters");
  }
#else  // !defined(USE_ITERATIONS)
  static void usage(const char *prog) {
    std::cerr << "Usage:\n"
              << "\t" << prog << " {dtrtri,strtri} <n>\n"
              << "\t" << prog << " {dpotrf,spotrf} <n>\n"
              << "\t" << prog << " {dgetrf,sgetrf} <m> <n>\n"
              << "\t" << prog << " {dgetrs,sgetrs,dgesv,sgesv} <n> <n_rhs>\n"
              << "\t" << prog << " {dgels,sgels} <n> <m> <n_rhs>\n";
  }

  static void get_iterations(int, const char *const *, int) {}
#endif // defined(USE_ITERATIONS)
};
} // namespace

int main(int argc, char **argv) {
  try {
    const cmdargs args(argc, argv);
    std::random_device rnd_dev;
    std::mt19937_64 engine{rnd_dev()};
    cusolverdn_handle handle{cusolverdn_create()};
    args.func(handle, args.params, engine);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
