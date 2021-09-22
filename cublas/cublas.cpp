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
        struct dgemm_caller {};
        template<>
        struct dgemm_caller<float> : func_obj<cublasSgemm> {};
        template<>
        struct dgemm_caller<double> : func_obj<cublasDgemm> {};

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

        template<typename Real, auto Func = dgemm_caller<Real>::value>
        void gemm_impl(
            std::size_t M,
            std::size_t N,
            std::size_t K,
            cublas_handle& handle,
            std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            Real alpha = 1.0;
            Real beta = 0.0;
            util::host_buffer<Real> a{ M * K };
            util::host_buffer<Real> b{ K * N };
            util::host_buffer<Real> c{ M * N };
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            smp.do_sample();
            util::device_buffer<Real> dev_a{ a.size() };
            util::device_buffer<Real> dev_b{ b.size() };
            util::device_buffer<Real> dev_c{ c.size() };
            auto res = cublasSetMatrix(
                M, K, sizeof(typename decltype(a)::value_type),
                a.get(), M,
                dev_a.get(), M);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error setting matrix A");
            res = cublasSetMatrix(
                K, N, sizeof(typename decltype(b)::value_type),
                b.get(), K,
                dev_b.get(), K);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error setting matrix B");
            res = cublasSetMatrix(
                M, N, sizeof(typename decltype(c)::value_type),
                c.get(), M,
                dev_c.get(), M);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error setting matrix C");

            smp.do_sample();
            res = Func(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                M, N, K, &alpha,
                dev_a.get(), M,
                dev_b.get(), K,
                &beta,
                dev_c.get(), M);
            handle_error(res);
            cudaDeviceSynchronize();

            smp.do_sample();
            res = cublasGetMatrix(
                M, K, sizeof(typename decltype(dev_a)::value_type),
                dev_a.get(), M,
                a.get(), M);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error getting matrix A");
            res = cublasGetMatrix(
                K, N, sizeof(typename decltype(dev_b)::value_type),
                dev_b.get(), K,
                b.get(), K);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error getting matrix B");
            res = cublasGetMatrix(
                M, N, sizeof(typename decltype(dev_c)::value_type),
                dev_c.get(), M,
                c.get(), M);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error getting matrix C");
        }
    }

    __attribute__((noinline)) void dgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        cublas_handle& handle,
        std::mt19937_64& engine)
    {
        detail::gemm_impl<double>(M, N, K, handle, engine);
    }

    __attribute__((noinline)) void sgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        cublas_handle& handle,
        std::mt19937_64& engine)
    {
        detail::gemm_impl<float>(M, N, K, handle, engine);
    }

    struct cmdargs
    {
        using work_func = decltype(&sgemm);
        std::size_t m = 0;
        std::size_t n = 0;
        std::size_t k = 0;
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

            util::to_scalar(argv[2], m);
            util::to_scalar(argv[3], n);
            util::to_scalar(argv[4], k);

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
        }

        void do_work(cublas_handle& handle, std::mt19937_64& engine) const
        {
            func(m, n, k, handle, engine);
        }

    private:
        void print_usage(const char* prog)
        {
            std::cerr << "Usage: " << prog << " {dgemm,sgemm} <m> <n> <k>\n";
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
    }
}
