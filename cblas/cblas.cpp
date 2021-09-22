#if defined V_USE_OPENBLAS
#include <cblas.h>
#elif defined V_USE_MKL
#include <mkl.h>
#endif

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

    namespace detail
    {
        template<auto Func>
        struct func_obj : std::integral_constant<decltype(Func), Func> {};

        template<typename>
        struct gemm_caller {};

        template<>
        struct gemm_caller<float> : func_obj<cblas_sgemm> {};

        template<>
        struct gemm_caller<double> : func_obj<cblas_dgemm> {};

        template<typename Real, CBLAS_TRANSPOSE Trans, auto Func = gemm_caller<Real>::value>
        void gemm_impl(std::size_t M, std::size_t N, std::size_t K, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            std::vector<Real> a(M * K);
            std::vector<Real> b(K * N);
            std::vector<Real> c(M * N);
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            smp.do_sample();
            Func(CblasRowMajor, CblasNoTrans, Trans,
                M, N, K, 1.0,
                a.data(), K,
                b.data(), N,
                0, c.data(), N
            );
        }
    }

    __attribute__((noinline))
        void dgemm_notrans(std::size_t M, std::size_t N, std::size_t K, std::mt19937_64& engine)
    {
        detail::gemm_impl<double, CblasNoTrans>(M, N, K, engine);
    }

    __attribute__((noinline))
        void dgemm(std::size_t M, std::size_t N, std::size_t K, std::mt19937_64& engine)
    {
        detail::gemm_impl<double, CblasTrans>(M, N, K, engine);
    }

    __attribute__((noinline))
        void sgemm_notrans(std::size_t M, std::size_t N, std::size_t K, std::mt19937_64& engine)
    {
        detail::gemm_impl<float, CblasNoTrans>(M, N, K, engine);
    }

    __attribute__((noinline))
        void sgemm(std::size_t M, std::size_t N, std::size_t K, std::mt19937_64& engine)
    {
        detail::gemm_impl<float, CblasTrans>(M, N, K, engine);
    }

    __attribute__((noinline))
        void dgemv(std::size_t M, std::size_t N, std::size_t, std::mt19937_64& engine)
    {
        std::uniform_real_distribution<double> dist{ 0.0, 1.0 };
        auto gen = [&]() { return dist(engine); };

        tp::sampler smp(g_tpr);
        std::vector<double> a(M * N);
        std::vector<double> x(N);
        std::vector<double> y(M);
        std::generate(a.begin(), a.end(), gen);

        smp.do_sample();
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
            M, N, 1.0,
            a.data(), N,
            x.data(), 1,
            0, y.data(), 1);
    }

    struct cmdparams
    {
        using work_func = decltype(&sgemm);
        std::size_t m = 0;
        std::size_t n = 0;
        std::size_t k = 0;
        work_func func = nullptr;

        cmdparams(int argc, const char* const* argv)
        {
            if (argc < 4)
            {
                print_usage(argv[0]);
                throw std::invalid_argument("Not enough arguments");
            }
            std::string op_type = argv[1];
            std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                [](unsigned char c)
                {
                    return std::tolower(c);
                });

            util::to_scalar(argv[2], m);
            util::to_scalar(argv[3], n);
            if (op_type == "dgemv")
                func = dgemv;
            else
            {
                if (argc < 5)
                {
                    print_usage(argv[0]);
                    throw std::invalid_argument(op_type.append(": not enough arguments"));
                }
                util::to_scalar(argv[4], k);
                if (op_type == "dgemm")
                    func = dgemm;
                else if (op_type == "dgemm_notrans")
                    func = dgemm_notrans;
                else if (op_type == "sgemm")
                    func = sgemm;
                else if (op_type == "sgemm_notrans")
                    func = sgemm_notrans;
                else
                {
                    print_usage(argv[0]);
                    throw std::invalid_argument(std::string("invalid work type: ").append(op_type));
                }
            }
            assert(func);
        }

        void do_work(std::mt19937_64& engine) const
        {
            func(m, n, k, engine);
        }

    private:
        void print_usage(const char* prog)
        {
            std::cerr << "Usage: " << prog
                << " {dgemm,dgemm_notrans,sgemm,sgemm_notrans,dgemv} <m> <n> <k>\n";
        }
    };
}

int main(int argc, char** argv)
{
    try
    {
        const cmdparams params(argc, argv);
        std::random_device rnd_dev;
        std::mt19937_64 engine{ rnd_dev() };
        params.do_work(engine);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}
