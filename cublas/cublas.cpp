#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <random>
#include <vector>

namespace
{
    tp::printer g_tpr;

    class cublas_handle
    {
        cublasHandle_t _handle;

    public:
        cublas_handle() :
            _handle(nullptr)
        {
            if (cublasStatus_t ret = cublasCreate(&_handle); ret != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error instantiating cuBLAS");
        }

        ~cublas_handle()
        {
            if (cublasStatus_t ret = cublasDestroy(_handle); ret != CUBLAS_STATUS_SUCCESS)
                std::cerr << "Error destroying cuBLAS\n";
        }

        cublasHandle_t get()
        {
            return _handle;
        }

        operator cublasHandle_t ()
        {
            return get();
        }
    };

    template<typename T>
    class device_buffer
    {
    public:
        using value_type = T;

    public:
        device_buffer(std::size_t size)
        {
            auto res = cudaMalloc(reinterpret_cast<void**>(&_ptr), size * sizeof(value_type));
            if (res != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(res));
        }

        ~device_buffer()
        {
            auto res = cudaFree(_ptr);
            if (res != cudaSuccess)
                std::cerr << "Error freeing device memory: " << cudaGetErrorString(res) << "\n";
        }

        const T* get() const
        {
            return _ptr;
        }

        T* get()
        {
            return _ptr;
        }

        operator const T* () const
        {
            return get();
        }

        operator T* ()
        {
            return get();
        }

    private:
        T* _ptr;
    };

    using work_func = void(*)(
        std::size_t,
        std::size_t,
        std::size_t,
        cublasHandle_t,
        std::mt19937_64&);

    namespace detail
    {
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

        template<typename Real, typename Func>
        void gemm_impl(
            Func func,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            cublasHandle_t handle,
            std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            Real alpha = 1;
            Real beta = 0;
            std::vector<Real> a(M * K);
            std::vector<Real> b(K * N);
            std::vector<Real> c(M * N);
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            smp.do_sample();
            device_buffer<Real> dev_a(a.size());
            device_buffer<Real> dev_b(b.size());
            device_buffer<Real> dev_c(c.size());
            auto res = cublasSetMatrix(
                M, K, sizeof(typename decltype(a)::value_type),
                a.data(), M,
                dev_a.get(), M);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error setting matrix A");
            res = cublasSetMatrix(
                K, N, sizeof(typename decltype(b)::value_type),
                b.data(), K,
                dev_b.get(), K);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error setting matrix B");
            res = cublasSetMatrix(
                M, N, sizeof(typename decltype(c)::value_type),
                c.data(), M,
                dev_c.get(), M);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error setting matrix C");

            smp.do_sample();
            res = func(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                M, N, K, &alpha,
                dev_a, M,
                dev_b, K,
                &beta,
                dev_c, M);
            handle_error(res);
            cudaDeviceSynchronize();

            smp.do_sample();
            res = cublasGetMatrix(
                M, K, sizeof(typename decltype(dev_a)::value_type),
                dev_a.get(), M,
                a.data(), M);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error getting matrix A");
            res = cublasGetMatrix(
                K, N, sizeof(typename decltype(dev_b)::value_type),
                dev_b.get(), K,
                b.data(), K);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error getting matrix B");
            res = cublasGetMatrix(
                M, N, sizeof(typename decltype(dev_c)::value_type),
                dev_c.get(), M,
                c.data(), M);
            if (res != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Error getting matrix C");
        }
    }

    __attribute__((noinline)) void dgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        cublasHandle_t handle,
        std::mt19937_64& engine)
    {
        detail::gemm_impl<double>(cublasDgemm, M, N, K, handle, engine);
    }

    __attribute__((noinline)) void sgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        cublasHandle_t handle,
        std::mt19937_64& engine)
    {
        detail::gemm_impl<float>(cublasSgemm, M, N, K, handle, engine);
    }

    struct cmdargs
    {
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

        void do_work(cublasHandle_t handle, std::mt19937_64& engine) const
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
        cmdargs args(argc, argv);
        std::random_device rnd_dev;
        std::mt19937_64 engine{ rnd_dev() };
        cublas_handle handle;
        args.do_work(handle, engine);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}
