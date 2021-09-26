#if defined V_USE_OPENBLAS
#include <lapacke.h>
#elif defined V_USE_MKL
#include <mkl.h>
#endif

#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>

#include <algorithm>
#include <cassert>
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
                throw std::runtime_error(
                    std::string("The solution could not be computed; info=").append(
                        std::to_string(res)));
            if (res < 0)
                throw std::runtime_error(std::string("Error during computation; info=").append(
                    std::to_string(res)));
        }

        template<auto Func>
        struct func_obj : std::integral_constant<decltype(Func), Func> {};

    #define DEFINE_CALLER(prefix) \
        template<typename> \
        struct prefix ## _caller {}; \
        template<> \
        struct prefix ## _caller<float> : func_obj<LAPACKE_s ## prefix ## _work> {}; \
        template<> \
        struct prefix ## _caller<double> : func_obj<LAPACKE_d ## prefix ## _work> {}

        DEFINE_CALLER(gesv);
        DEFINE_CALLER(gels);
        DEFINE_CALLER(getri);
        DEFINE_CALLER(getrf);
        DEFINE_CALLER(tptri);
        DEFINE_CALLER(trtri);

        template<typename Real>
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
            int res = gesv_caller<Real>::value(
                LAPACK_COL_MAJOR, N, Nrhs, a.data(), N, ipiv.data(), b.data(), N);
            handle_error(res);
        }

        template<typename Real>
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
            Real workspace_query;
            int res = gels_caller<Real>::value(LAPACK_COL_MAJOR,
                'N', M, N, Nrhs, a.data(), M, b.data(), std::max(N, M), &workspace_query, -1);
            handle_error(res);

            smp.do_sample();
            std::vector<Real> work(static_cast<std::uint64_t>(workspace_query));
            res = gels_caller<Real>::value(LAPACK_COL_MAJOR,
                'N', M, N, Nrhs, a.data(), M, b.data(), std::max(N, M), work.data(), work.size());
            handle_error(res);
        }

        template<typename Real>
        void getri_impl(std::size_t N, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            std::vector<Real> a(N * N);
            std::vector<lapack_int> ipiv(N);
            std::generate(a.begin(), a.end(), gen);

            smp.do_sample();
            int res = getrf_caller<Real>::value(
                LAPACK_COL_MAJOR, N, N, a.data(), N, ipiv.data());
            handle_error(res);

            smp.do_sample();
            Real workspace_query;
            res = getri_caller<Real>::value(LAPACK_COL_MAJOR,
                N, a.data(), N, ipiv.data(), &workspace_query, -1);
            handle_error(res);

            smp.do_sample();
            std::vector<Real> work(static_cast<std::uint64_t>(workspace_query));
            res = getri_caller<Real>::value(
                LAPACK_COL_MAJOR, N, a.data(), N, ipiv.data(), work.data(), work.size());
            handle_error(res);
        }

        template<typename Real>
        void getrf_impl(std::size_t M, std::size_t N, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            std::vector<Real> a(M * N);
            std::vector<lapack_int> ipiv(std::min(M, N));
            std::generate(a.begin(), a.end(), gen);

            smp.do_sample();
            int res = getrf_caller<Real>::value(
                LAPACK_COL_MAJOR, M, N, a.data(), M, ipiv.data());
            handle_error(res);
        }

        template<typename Real>
        void tptri_impl(std::size_t N, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 1.0, 2.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            std::vector<Real> a_packed(N * (N + 1) / 2);
            std::generate(a_packed.begin(), a_packed.end(), gen);

            smp.do_sample();
            int res = tptri_caller<Real>::value(
                LAPACK_COL_MAJOR, 'L', 'N', N, a_packed.data());
            handle_error(res);
        }

        template<typename Real>
        void trtri_impl(std::size_t N, std::mt19937_64& engine)
        {
            // upper triangular in column-major is lower triangular in row-major
            auto fill_upper_triangular = [](auto from, auto to, std::size_t ld, auto gen)
            {
                for (auto [it, nnz] = std::pair{ from, ld }; it < to; it += ld + 1, nnz--)
                    for (auto entry = it; entry < it + nnz; entry++)
                        *entry = gen();
            };

            std::uniform_real_distribution<Real> dist{ 1.0, 2.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);
            std::vector<Real> a(N * N);
            fill_upper_triangular(a.begin(), a.end(), N, gen);

            smp.do_sample();
            int res = trtri_caller<Real>::value(
                LAPACK_COL_MAJOR, 'L', 'N', N, a.data(), N);
            handle_error(res);
        }
    }

    using work_func = void(*)(
        std::size_t,
        std::size_t,
        std::size_t,
        std::mt19937_64&);

    __attribute__((noinline))
        void dgesv(std::size_t, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
    {
        detail::gesv_impl<double>(N, Nrhs, engine);
    }

    __attribute__((noinline))
        void sgesv(std::size_t, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
    {
        detail::gesv_impl<float>(N, Nrhs, engine);
    }

    __attribute__((noinline))
        void dgels(std::size_t M, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
    {
        detail::gels_impl<double>(M, N, Nrhs, engine);
    }

    __attribute__((noinline))
        void sgels(std::size_t M, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
    {
        detail::gels_impl<float>(M, N, Nrhs, engine);
    }

    __attribute__((noinline))
        void dgetri(std::size_t, std::size_t N, std::size_t, std::mt19937_64& engine)
    {
        detail::getri_impl<double>(N, engine);
    }

    __attribute__((noinline))
        void sgetri(std::size_t, std::size_t N, std::size_t, std::mt19937_64& engine)
    {
        detail::getri_impl<float>(N, engine);
    }

    __attribute__((noinline))
        void dgetrf(std::size_t M, std::size_t N, std::size_t, std::mt19937_64& engine)
    {
        detail::getrf_impl<double>(M, N, engine);
    }

    __attribute__((noinline))
        void sgetrf(std::size_t M, std::size_t N, std::size_t, std::mt19937_64& engine)
    {
        detail::getrf_impl<float>(M, N, engine);
    }

    __attribute__((noinline))
        void dtptri(std::size_t, std::size_t N, std::size_t, std::mt19937_64& engine)
    {
        detail::tptri_impl<double>(N, engine);
    }

    __attribute__((noinline))
        void stptri(std::size_t, std::size_t N, std::size_t, std::mt19937_64& engine)
    {
        detail::tptri_impl<float>(N, engine);
    }

    __attribute__((noinline))
        void dtrtri(std::size_t, std::size_t N, std::size_t, std::mt19937_64& engine)
    {
        detail::trtri_impl<double>(N, engine);
    }

    __attribute__((noinline))
        void strtri(std::size_t, std::size_t N, std::size_t, std::mt19937_64& engine)
    {
        detail::trtri_impl<float>(N, engine);
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
            else if (is_inversion(func))
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

        void do_work(std::mt19937_64& engine) const
        {
            func(m, n, nrhs, engine);
        }

    private:
        bool is_inversion(work_func func)
        {
            return func == dgetri || func == sgetri ||
                func == dtptri || func == stptri ||
                func == dtrtri || func == strtri;
        }

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
            if (str == "dtptri")
                return dtptri;
            if (str == "stptri")
                return stptri;
            if (str == "dtrtri")
                return dtrtri;
            if (str == "strtri")
                return strtri;
            return nullptr;
        }

        void print_usage(const char* prog)
        {
            std::cerr << "Usage:\n"
                << "\t" << prog << " {dgesv,sgesv} <n> <nrhs>\n"
                << "\t" << prog << " {dgels,sgels} <m> <n> <nrhs>\n"
                << "\t" << prog << " {dgetri,sgetri} <n>\n"
                << "\t" << prog << " {dtptri,stptri,dtrtri,strtri} <n>\n"
                << "\t" << prog << " {dgetrf,sgetrf} <m> <n>\n";
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
