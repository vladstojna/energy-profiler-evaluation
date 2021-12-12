#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>
#include <util/unique_handle.hpp>

#include <cublasXt.h>

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

#define NO_INLINE __attribute__((noinline))
#define NO_CLONE __attribute__((noclone))

namespace
{
    tp::printer g_tpr;

    template<auto Func>
    struct func_obj : std::integral_constant<decltype(Func), Func> {};

    cublasXtHandle_t cublasxt_create()
    {
        cublasXtHandle_t handle;
        if (auto ret = cublasXtCreate(&handle); ret != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error instantiating cuBLASXt");
        return handle;
    }

    void cublasxt_destroy(cublasXtHandle_t handle)
    {
        if (auto ret = cublasXtDestroy(handle); ret != CUBLAS_STATUS_SUCCESS)
            std::cerr << "Error destroying cuBLASXt\n";
    }

    using cublasxt_handle = util::unique_handle<cublasXtHandle_t, func_obj<cublasxt_destroy>>;

    namespace detail
    {
        template<typename>
        struct gemm_caller {};
        template<>
        struct gemm_caller<float> : func_obj<cublasXtSgemm> {};
        template<>
        struct gemm_caller<double> : func_obj<cublasXtDgemm> {};

        void handle_error(cublasStatus_t status)
        {
            if (status == CUBLAS_STATUS_SUCCESS)
                return;
            switch (status)
            {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                throw std::runtime_error("Library not initialized");
            case CUBLAS_STATUS_INVALID_VALUE:
                throw std::runtime_error("Parameters m, n or k less than 0");
            case CUBLAS_STATUS_EXECUTION_FAILED:
                throw std::runtime_error("The function failed to launch on the GPU");
            default:
                throw std::runtime_error("Some other error occurred");
            }
        }

        template<typename Real>
        NO_INLINE NO_CLONE void gemm_compute(
            cublasxt_handle& handle,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            const Real* alpha,
            const Real* a,
            const Real* b,
            const Real* beta,
            Real* c)
        {
            tp::sampler smp(g_tpr);
            auto res = gemm_caller<Real>::value(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, alpha, a, M, b, K, beta, c, M);
            handle_error(res);
        }

        template<typename Real>
        void gemm_impl(
            std::size_t M,
            std::size_t N,
            std::size_t K,
            cublasxt_handle& handle,
            std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            constexpr Real alpha = 1.0;
            constexpr Real beta = 0.0;
            std::vector<Real> a(M * K);
            std::vector<Real> b(K * N);
            std::vector<Real> c(M * N);
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);
            gemm_compute(
                handle, M, N, K, &alpha, a.data(), b.data(), &beta, c.data());
        }
    }

    NO_INLINE void dgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        cublasxt_handle& handle,
        std::mt19937_64& engine)
    {
        detail::gemm_impl<double>(M, N, K, handle, engine);
    }

    NO_INLINE void sgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        cublasxt_handle& handle,
        std::mt19937_64& engine)
    {
        detail::gemm_impl<float>(M, N, K, handle, engine);
    }

    void set_first_device(cublasxt_handle& handle)
    {
        int device_ids[] = { 0 };
        if (auto res = cublasXtDeviceSelect(handle, 1, device_ids); res != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasXtDeviceSelect error");
    }

    void set_block_dim(cublasxt_handle& handle, int block_dim)
    {
        if (auto res = cublasXtSetBlockDim(handle, block_dim); res != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasXtSetBlockDim error");
    }

    struct cmdargs
    {
        using work_func = decltype(&sgemm);
        std::size_t m = 0;
        std::size_t n = 0;
        std::size_t k = 0;
        int block_dim = 0;
        work_func func = nullptr;

        cmdargs(int argc, const char* const* argv)
        {
            if (argc < 5)
            {
                print_usage(argv[0]);
                throw std::invalid_argument("Not enough arguments");
            }
            std::string op_type = argv[1];
            std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                [](unsigned char c) { return std::tolower(c); });
            if (op_type == "dgemm")
                func = dgemm;
            else if (op_type == "sgemm")
                func = sgemm;
            else
            {
                print_usage(argv[0]);
                throw std::invalid_argument(std::string("invalid work type: ").append(argv[1]));
            }
            assert(func);

            util::to_scalar(argv[2], m);
            assert_positive(m, "m");
            util::to_scalar(argv[3], n);
            assert_positive(n, "n");
            util::to_scalar(argv[4], k);
            assert_positive(k, "k");

            if (argc >= 6)
            {
                util::to_scalar(argv[5], block_dim);
                if (block_dim <= 0)
                    throw std::invalid_argument(
                        std::string("block dimension must be positive, found: ")
                        .append(std::to_string(block_dim)));
            }
        }

        void do_work(cublasxt_handle& handle, std::mt19937_64& engine) const
        {
            func(m, n, k, handle, engine);
        }

    private:
        void assert_positive(std::size_t x, std::string name)
        {
            assert(x);
            if (!x)
                throw std::invalid_argument(std::move(name.append(" must be greater than 0")));
        }

        void print_usage(const char* prog)
        {
            std::cerr << "Usage: " << prog << " {dgemm,sgemm} <m> <n> <k> <block_dim>\n";
        }
    };
}

int main(int argc, char** argv)
{
    try
    {
        const cmdargs args(argc, argv);
        std::random_device rnd_dev;
        std::mt19937_64 engine{ rnd_dev() };
        cublasxt_handle handle{ cublasxt_create() };
        set_first_device(handle);
        if (args.block_dim)
            set_block_dim(handle, args.block_dim);
        args.do_work(handle, engine);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
