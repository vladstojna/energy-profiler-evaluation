#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>
#include <util/unique_handle.hpp>

#include <cublasXt.h>

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

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

        template<typename Real, auto Func = gemm_caller<Real>::value>
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
            std::vector<Real> a(M * K);
            std::vector<Real> b(K * N);
            std::vector<Real> c(M * N);
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            Real alpha = 1.0;
            Real beta = 0.0;

            smp.do_sample();
            cublasStatus_t res = Func(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                M, N, K, &alpha,
                a.data(), M,
                b.data(), K,
                &beta, c.data(), M);
            handle_error(res);
        }
    }

    __attribute__((noinline)) void dgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        cublasxt_handle& handle,
        std::mt19937_64& engine)
    {
        detail::gemm_impl<double>(M, N, K, handle, engine);
    }

    __attribute__((noinline)) void sgemm(
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

            if (argc >= 6)
            {
                util::to_scalar(argv[5], block_dim);
                if (block_dim <= 0)
                    throw std::invalid_argument(
                        std::string("block dimension must be positive, found: ")
                        .append(std::to_string(block_dim))
                    );
            }
        }

        void do_work(cublasxt_handle& handle, std::mt19937_64& engine) const
        {
            func(m, n, k, handle, engine);
        }

    private:
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
    }
}
