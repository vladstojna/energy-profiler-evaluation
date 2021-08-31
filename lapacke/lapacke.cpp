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
    namespace detail
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

        template<typename Real, typename Func>
        void gesv_impl(
            Func func,
            std::size_t N,
            std::size_t Nrhs,
            std::mt19937_64& engine,
            std::uniform_real_distribution<double>& dist)
        {
            std::vector<Real> a(N * N);
            std::vector<Real> b(N * Nrhs);
            std::vector<lapack_int> ipiv(N);

            auto gen = get_gen(engine, dist);
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            tp::printer tpr;
            int res = func(LAPACK_ROW_MAJOR, N, Nrhs, a.data(), N, ipiv.data(), b.data(), Nrhs);
            handle_error(res);
        }

        template<typename Real, typename Func>
        void gels_impl(
            Func func,
            std::size_t M,
            std::size_t N,
            std::size_t Nrhs,
            std::mt19937_64& engine,
            std::uniform_real_distribution<double>& dist)
        {
            std::vector<Real> a(M * N);
            std::vector<Real> b(std::max(N, M) * Nrhs);

            auto gen = get_gen(engine, dist);
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            tp::printer tpr;
            int res = func(LAPACK_ROW_MAJOR, 'N', M, N, Nrhs, a.data(), N, b.data(), Nrhs);
            handle_error(res);
        }
    }

    using work_func = void(*)(
        std::size_t,
        std::size_t,
        std::size_t,
        std::mt19937_64&,
        std::uniform_real_distribution<double>&);

    __attribute__((noinline)) void dgesv(
        std::size_t,
        std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        detail::gesv_impl<double>(LAPACKE_dgesv, N, Nrhs, engine, dist);
    }

    __attribute__((noinline)) void sgesv(
        std::size_t,
        std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        detail::gesv_impl<float>(LAPACKE_sgesv, N, Nrhs, engine, dist);
    }

    __attribute__((noinline)) void dgels(
        std::size_t M,
        std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        detail::gels_impl<double>(LAPACKE_dgels, M, N, Nrhs, engine, dist);
    }

    __attribute__((noinline)) void sgels(
        std::size_t M,
        std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine,
        std::uniform_real_distribution<double>& dist)
    {
        detail::gels_impl<float>(LAPACKE_sgels, M, N, Nrhs, engine, dist);
    }

    class cmdparams
    {
        std::size_t m = 0;
        std::size_t n = 0;
        std::size_t nrhs = 0;
        work_func func = nullptr;

    public:
        cmdparams(int argc, const char* const* argv)
        {
            if (argc < 2)
            {
                print_usage(argv[0]);
                throw std::invalid_argument("Too few arguments");
            }
            std::string op_type = argv[1];
            std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                [](unsigned char c)
                {
                    return std::tolower(c);
                });

            func = get_work_func(op_type);
            if (func == dgels || func == sgels)
            {
                if (argc < 5)
                {
                    print_usage(argv[0]);
                    throw std::invalid_argument(op_type.append(": Too few arguments"));
                }
                util::to_scalar(argv[2], m);
                util::to_scalar(argv[3], n);
                util::to_scalar(argv[4], nrhs);
            }
            else if (func == dgesv || func == sgesv)
            {
                if (argc < 4)
                {
                    print_usage(argv[0]);
                    throw std::invalid_argument(op_type.append(": Too few arguments"));
                }
                util::to_scalar(argv[2], n);
                util::to_scalar(argv[3], nrhs);
            }
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
            func(m, n, nrhs, engine, dist);
        }

    private:
        work_func get_work_func(const std::string& str)
        {
            if (str == "dgels")
                return dgels;
            if (str == "dgesv")
                return dgesv;
            if (str == "sgels")
                return sgels;
            if (str == "sgesv")
                return sgesv;
            return nullptr;
        }

        void print_usage(const char* prog)
        {
            std::cerr << "Usage:\n"
                << "\t" << prog << " {dgesv,sgesv} <n> <nrhs>\n"
                << "\t" << prog << " {dgels,sgels} <m> <n> <nrhs>\n";
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
