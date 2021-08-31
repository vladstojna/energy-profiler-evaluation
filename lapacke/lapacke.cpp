#if defined V_USE_BASELINE
#include <lapacke.h>
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

    void handle_error(int res)
    {
        if (res > 0)
            throw std::logic_error("The solution could not be computed");
        if (res < 0)
            throw std::runtime_error("Error during computation");
    }

    __attribute__((noinline)) void dgesv(std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        std::vector<double> a(N * N);
        std::vector<double> b(N * Nrhs);
        std::vector<lapack_int> ipiv(N);

        auto gen = get_gen(engine, dist);
        std::generate(a.begin(), a.end(), gen);
        std::generate(b.begin(), b.end(), gen);

        tp::printer tpr;
        int res = LAPACKE_dgesv(
            LAPACK_ROW_MAJOR, N, Nrhs,
            a.data(), N,
            ipiv.data(),
            b.data(), Nrhs);

        handle_error(res);
    }

    __attribute__((noinline)) void sgesv(std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        std::vector<float> a(N * N);
        std::vector<float> b(N * Nrhs);
        std::vector<lapack_int> ipiv(N);

        auto gen = get_gen(engine, dist);
        std::generate(a.begin(), a.end(), gen);
        std::generate(b.begin(), b.end(), gen);

        tp::printer tpr;
        int res = LAPACKE_sgesv(
            LAPACK_ROW_MAJOR, N, Nrhs,
            a.data(), N,
            ipiv.data(),
            b.data(), Nrhs);

        handle_error(res);
    }

    using work_func = void(*)(
        std::size_t,
        std::size_t,
        std::mt19937_64&,
        std::uniform_real_distribution<double>&);

    class cmdparams
    {
        std::size_t n = 0;
        std::size_t nrhs = 0;
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

            util::to_scalar(argv[2], n);
            util::to_scalar(argv[3], nrhs);
            if (op_type == "dgesv")
                func = dgesv;
            else if (op_type == "sgesv")
                func = sgesv;
            else
            {
                print_usage(argv[0]);
                throw std::invalid_argument(std::string("invalid work type: ").append(op_type));
            }
            assert(func);
        }

        void do_work(
            std::mt19937_64& engine,
            std::uniform_real_distribution<double>& dist)
        {
            func(n, nrhs, engine, dist);
        }

    private:
        void print_usage(const char* prog)
        {
            std::cerr << "Usage: " << prog << " {dgesv,sgesv} <n> <nrhs>\n";
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
