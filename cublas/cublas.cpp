#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>
#include <util/cuda_utils.hpp>
#include <util/unique_handle.hpp>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <random>

#define NO_INLINE __attribute__((noinline))
#define NO_CLONE __attribute__((noclone))

namespace
{
    tp::printer g_tpr;

    template<auto Func>
    struct func_obj : std::integral_constant<decltype(Func), Func> {};

    cublasHandle_t cublas_create()
    {
        cublasHandle_t handle;
        auto status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Error creating cuBLAS");
        return handle;
    }

    void cublas_destroy(cublasHandle_t handle)
    {
        auto status = cublasDestroy(handle);
        if (status != CUBLAS_STATUS_SUCCESS)
            std::cerr << "Error destroying cuBLAS\n";
    }

    using cublas_handle = util::unique_handle<cublasHandle_t, func_obj<cublas_destroy>>;

    namespace detail
    {
        template<typename>
        struct gemm_caller {};
        template<>
        struct gemm_caller<float> : func_obj<cublasSgemm> {};
        template<>
        struct gemm_caller<double> : func_obj<cublasDgemm> {};

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
            case CUBLAS_STATUS_ARCH_MISMATCH:
                throw std::runtime_error("The device does not support math in half precision");
            default:
                throw std::runtime_error("Some other error occurred");
            }
        }

        template<typename Real>
        NO_INLINE NO_CLONE void gemm_compute(
            cublas_handle& handle,
            std::size_t iters,
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
            for (decltype(iters) i = 0; i < iters; ++i)
            {
                auto res = gemm_caller<Real>::value(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K, alpha, a, M, b, K, beta, c, M);
                handle_error(res);
                cudaDeviceSynchronize();
            }
        }

        template<typename Real>
        void gemm_impl(
            std::size_t M,
            std::size_t N,
            std::size_t K,
            std::size_t iters,
            cublas_handle& handle,
            std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            constexpr Real alpha = 1.0;
            constexpr Real beta = 0.0;
            util::buffer<Real> a{ M * K };
            util::buffer<Real> b{ K * N };
            util::buffer<Real> c{ M * N };
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);
            smp.do_sample();
            util::device_buffer dev_a{ a.begin(), a.end() };
            util::device_buffer dev_b{ b.begin(), b.end() };
            util::device_buffer<Real> dev_c{ c.size() };
            gemm_compute(handle, iters, M, N, K,
                &alpha, dev_a.get(), dev_b.get(), &beta, dev_c.get());
            util::copy(dev_c, dev_c.size(), c.begin());
        }
    }

    NO_INLINE void dgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        std::size_t iters,
        cublas_handle& handle,
        std::mt19937_64& engine)
    {
        detail::gemm_impl<double>(M, N, K, iters, handle, engine);
    }

    NO_INLINE void sgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        std::size_t iters,
        cublas_handle& handle,
        std::mt19937_64& engine)
    {
        detail::gemm_impl<float>(M, N, K, iters, handle, engine);
    }

    struct cmdargs
    {
        using work_func = decltype(&sgemm);
        std::size_t m = 0;
        std::size_t n = 0;
        std::size_t k = 0;
        std::size_t iters = 1;
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

            if (argc > 5)
            {
                util::to_scalar(argv[5], iters);
                assert_positive(iters, "iters");
            }
        }

        void do_work(cublas_handle& handle, std::mt19937_64& engine) const
        {
            func(m, n, k, iters, handle, engine);
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
            std::cerr << "Usage: " << prog << " {dgemm,sgemm} <m> <n> <k> <iters>\n";
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
        cublas_handle handle{ cublas_create() };
        args.do_work(handle, engine);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
