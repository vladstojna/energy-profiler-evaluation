#if defined V_USE_OPENBLAS
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
    tp::printer g_tpr;

    namespace detail
    {
        void handle_error(int res)
        {
            if (res > 0)
                throw std::logic_error("The solution could not be computed");
            if (res < 0)
                throw std::runtime_error("Error during computation");
        }

        template<auto Func>
        struct func_obj : std::integral_constant<decltype(Func), Func> {};

        template<typename T>
        struct gesv_caller {};
        template<>
        struct gesv_caller<float> : func_obj<LAPACKE_sgesv> {};
        template<>
        struct gesv_caller<double> : func_obj<LAPACKE_dgesv> {};

        template<typename T>
        struct gels_caller {};
        template<>
        struct gels_caller<float> : func_obj<LAPACKE_sgels> {};
        template<>
        struct gels_caller<double> : func_obj<LAPACKE_dgels> {};

        template<typename T>
        struct getri_caller {};
        template<>
        struct getri_caller<float> : func_obj<LAPACKE_sgetri> {};
        template<>
        struct getri_caller<double> : func_obj<LAPACKE_dgetri> {};

        template<typename T>
        struct getrf_caller {};
        template<>
        struct getrf_caller<float> : func_obj<LAPACKE_sgetrf> {};
        template<>
        struct getrf_caller<double> : func_obj<LAPACKE_dgetrf> {};

        template<typename Real, auto Func = gesv_caller<Real>::value>
        void gesv_impl(std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            std::vector<Real> a(N * N);
            std::vector<Real> b(N * Nrhs);
            std::vector<lapack_int> ipiv(N);

            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            smp.do_sample();
            int res = Func(LAPACK_ROW_MAJOR, N, Nrhs, a.data(), N, ipiv.data(), b.data(), Nrhs);
            handle_error(res);
        }

        template<typename Real, auto Func = gels_caller<Real>::value>
        void gels_impl(std::size_t M, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            std::vector<Real> a(M * N);
            std::vector<Real> b(std::max(N, M) * Nrhs);

            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            smp.do_sample();
            int res = Func(LAPACK_ROW_MAJOR, 'N', M, N, Nrhs, a.data(), N, b.data(), Nrhs);
            handle_error(res);
        }

        template<typename Real,
            auto Ffact = getrf_caller<Real>::value,
            auto Finv = getri_caller<Real>::value
        > void getri_impl(std::size_t N, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            std::vector<Real> a(N * N);
            std::vector<lapack_int> ipiv(N);
            std::generate(a.begin(), a.end(), gen);

            smp.do_sample();
            int res = Ffact(LAPACK_ROW_MAJOR, N, N, a.data(), N, ipiv.data());
            handle_error(res);

            smp.do_sample();
            res = Finv(LAPACK_ROW_MAJOR, N, a.data(), N, ipiv.data());
            handle_error(res);
        }

        template<typename Real, auto Func = getrf_caller<Real>::value>
        void getrf_impl(std::size_t M, std::size_t N, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            std::vector<Real> a(M * N);
            std::vector<lapack_int> ipiv(std::min(M, N));
            std::generate(a.begin(), a.end(), gen);

            smp.do_sample();
            int res = Func(LAPACK_ROW_MAJOR, M, N, a.data(), N, ipiv.data());
            handle_error(res);
        }
    }

    using work_func = void(*)(
        std::size_t,
        std::size_t,
        std::size_t,
        std::mt19937_64&);

    __attribute__((noinline)) void dgesv(
        std::size_t,
        std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine)
    {
        detail::gesv_impl<double>(N, Nrhs, engine);
    }

    __attribute__((noinline)) void sgesv(
        std::size_t,
        std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine)
    {
        detail::gesv_impl<float>(N, Nrhs, engine);
    }

    __attribute__((noinline)) void dgels(
        std::size_t M,
        std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine)
    {
        detail::gels_impl<double>(M, N, Nrhs, engine);
    }

    __attribute__((noinline)) void sgels(
        std::size_t M,
        std::size_t N,
        std::size_t Nrhs,
        std::mt19937_64& engine)
    {
        detail::gels_impl<float>(M, N, Nrhs, engine);
    }

    __attribute__((noinline)) void dgetri(
        std::size_t,
        std::size_t N,
        std::size_t,
        std::mt19937_64& engine)
    {
        detail::getri_impl<double>(N, engine);
    }

    __attribute__((noinline)) void sgetri(
        std::size_t,
        std::size_t N,
        std::size_t,
        std::mt19937_64& engine)
    {
        detail::getri_impl<float>(N, engine);
    }

    __attribute__((noinline)) void dgetrf(
        std::size_t M,
        std::size_t N,
        std::size_t,
        std::mt19937_64& engine)
    {
        detail::getrf_impl<double>(M, N, engine);
    }

    __attribute__((noinline)) void sgetrf(
        std::size_t M,
        std::size_t N,
        std::size_t,
        std::mt19937_64& engine)
    {
        detail::getrf_impl<float>(M, N, engine);
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
            else if (func == dgetri || func == sgetri)
            {
                if (argc < 3)
                {
                    print_usage(argv[0]);
                    throw std::invalid_argument(op_type.append(": Too few arguments"));
                }
                util::to_scalar(argv[2], n);
            }
            else if (func == dgetrf || func == sgetrf)
            {
                if (argc < 4)
                {
                    print_usage(argv[0]);
                    throw std::invalid_argument(op_type.append(": Too few arguments"));
                }
                util::to_scalar(argv[2], m);
                util::to_scalar(argv[3], n);
            }
            else
            {
                print_usage(argv[0]);
                throw std::invalid_argument(std::string("invalid work type: ").append(op_type));
            }
            assert(func);
        }

        void do_work(std::mt19937_64& engine)
        {
            func(m, n, nrhs, engine);
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
            if (str == "dgetri")
                return dgetri;
            if (str == "sgetri")
                return sgetri;
            if (str == "dgetrf")
                return dgetrf;
            if (str == "sgetrf")
                return sgetrf;
            return nullptr;
        }

        void print_usage(const char* prog)
        {
            std::cerr << "Usage:\n"
                << "\t" << prog << " {dgesv,sgesv} <n> <nrhs>\n"
                << "\t" << prog << " {dgels,sgels} <m> <n> <nrhs>\n"
                << "\t" << prog << " {dgetri,sgetri} <n>\n"
                << "\t" << prog << " {dgetrf,sgetrf} <m> <n>\n";
        }
    };
}

int main(int argc, char** argv)
{
    std::random_device rnd_dev;
    std::mt19937_64 engine{ rnd_dev() };
    try
    {
        cmdparams params(argc, argv);
        params.do_work(engine);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}
