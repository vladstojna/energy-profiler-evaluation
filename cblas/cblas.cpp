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
    auto get_gen(std::mt19937_64& engine, std::uniform_real_distribution<double>& dist)
    {
        return [&]()
        {
            return dist(engine);
        };
    }

    __attribute__((noinline)) void dgemm_no_transpose(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        auto gen = get_gen(engine, dist);
        std::vector<double> a(M * K);
        std::vector<double> b(K * N);
        std::vector<double> c(M * N);
        std::generate(a.begin(), a.end(), gen);
        std::generate(b.begin(), b.end(), gen);

        tp::printer tpr;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, 1.0,
            a.data(), K,
            b.data(), N,
            0, c.data(), N
        );
    }

    __attribute__((noinline)) void dgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        std::vector<double> a(M * K);
        std::vector<double> b(K * N);
        std::vector<double> c(M * N);
        auto gen = get_gen(engine, dist);
        std::generate(a.begin(), a.end(), gen);
        std::generate(b.begin(), b.end(), gen);

        tp::printer tpr;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, K, 1.0,
            a.data(), K,
            b.data(), K,
            0, c.data(), N
        );
    }

    __attribute__((noinline)) void dgemv(
        std::size_t M,
        std::size_t N,
        std::size_t,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        std::vector<double> a(M * N);
        std::vector<double> x(N);
        std::vector<double> y(M);
        auto gen = get_gen(engine, dist);
        std::generate(a.begin(), a.end(), gen);

        tp::printer tpr;
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
            M, N, 1.0,
            a.data(), N,
            x.data(), 1,
            0, y.data(), 1);
    }

    __attribute__((noinline)) void sgemm(
        std::size_t M,
        std::size_t N,
        std::size_t K,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        std::vector<float> a(M * K);
        std::vector<float> b(K * N);
        std::vector<float> c(M * N);
        auto gen = get_gen(engine, dist);
        std::generate(a.begin(), a.end(), gen);
        std::generate(b.begin(), b.end(), gen);

        tp::printer tpr;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, K, 1.0,
            a.data(), K,
            b.data(), K,
            0, c.data(), N
        );
    }

    using work_func = void(*)(
        std::size_t,
        std::size_t,
        std::size_t,
        std::mt19937_64&,
        std::uniform_real_distribution<double>&);

    class cmdparams
    {
        std::size_t m = 0;
        std::size_t n = 0;
        std::size_t k = 0;
        work_func func = nullptr;

    public:
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
                    func = dgemm_no_transpose;
                else if (op_type == "sgemm")
                    func = sgemm;
                else
                {
                    print_usage(argv[0]);
                    throw std::invalid_argument(std::string("invalid work type: ").append(op_type));
                }
            }
            assert(func);
        }

        void do_work(
            std::mt19937_64& engine,
            std::uniform_real_distribution<double>& dist)
        {
            func(m, n, k, engine, dist);
        }

    private:
        void print_usage(const char* prog)
        {
            std::cerr << "Usage: " << prog
                << " {dgemm,dgemm_notrans,sgemm,dgemv} <m> <n> <k>\n";
        }
    };
}

int main(int argc, char** argv)
{
    std::random_device rnd_dev;
    std::mt19937_64 engine{ rnd_dev() };
    std::uniform_real_distribution dist{ 0.0, 1.0 };
    try
    {
        cmdparams params(argc, argv);
        params.do_work(engine, dist);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}
