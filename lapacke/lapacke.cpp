#if defined V_USE_OPENBLAS
#include <lapacke.h>
#elif defined V_USE_MKL
#include <mkl.h>
#endif

#include <timeprinter/printer.hpp>
#include <util/buffer.hpp>
#include <util/to_scalar.hpp>

#include <algorithm>
#include <cassert>
#include <random>

#define NO_INLINE __attribute__((noinline))

namespace
{
    tp::printer g_tpr;

#if defined(USE_ITERATIONS)
    struct compute_params
    {
        std::size_t N = 0;
        std::size_t M = 0;
        std::size_t Nrhs = 0;
        std::size_t iters = 0;
    };
#else
    struct compute_params
    {
        std::size_t N = 0;
        std::size_t M = 0;
        std::size_t Nrhs = 0;
    };
#endif // USE_ITERATIONS

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
        DEFINE_CALLER(getrs);
        DEFINE_CALLER(tptri);
        DEFINE_CALLER(trtri);
        DEFINE_CALLER(potrf);

        template<typename It, typename Gen>
        void fill_upper_triangular(It from, It to, std::size_t ld, Gen gen)
        {
            for (auto [it, nnz] = std::pair{ from, ld }; it < to; it += ld + 1, nnz--)
                for (auto entry = it; entry < it + nnz; entry++)
                    *entry = gen();
        }

        template<typename Real>
        util::buffer<Real> upper_dd_matrix(std::size_t N, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            util::buffer<Real> a{ N * N };
            std::fill(a.begin(), a.end(), Real{});
            fill_upper_triangular(a.begin(), a.end(), N, gen);
            smp.do_sample();
            {
                // compute A = A + rand(N, 2N) * Identity(N, N)
                // to guarantee that the matrix is diagonally dominant
                std::uniform_real_distribution<Real> dist{
                    static_cast<Real>(N),
                    static_cast<Real>(2 * N)
                };
                for (auto [it, x] = std::pair{ a.begin(), 0 }; it < a.end(); it += N, ++x)
                    *(it + x) += dist(engine);
            }
            return a;
        }

        namespace compute
        {
            template<typename Real>
            NO_INLINE void gesv(
                Real* a,
                Real* b,
                lapack_int* ipiv,
                std::size_t N,
                std::size_t Nrhs)
            {
                tp::sampler smp(g_tpr);
                int res = gesv_caller<Real>::value(
                    LAPACK_COL_MAJOR, N, Nrhs, a, N, ipiv, b, N);
                handle_error(res);
            }

            template<typename Real>
            NO_INLINE void gels(
                Real* a,
                Real* b,
                std::size_t M,
                std::size_t N,
                std::size_t Nrhs)
            {
                tp::sampler smp(g_tpr);
                Real workspace_query;
                int res = gels_caller<Real>::value(LAPACK_COL_MAJOR,
                    'N', M, N, Nrhs, a, M, b, std::max(N, M), &workspace_query, -1);
                handle_error(res);
                smp.do_sample();
                util::buffer<Real> work{ static_cast<std::uint64_t>(workspace_query) };
                res = gels_caller<Real>::value(LAPACK_COL_MAJOR,
                    'N', M, N, Nrhs, a, M, b, std::max(N, M), work.get(), work.size());
                handle_error(res);
            }

            template<typename Real>
            NO_INLINE void getrf(
                Real* a,
                lapack_int* ipiv,
                std::size_t M,
                std::size_t N)
            {
                tp::sampler smp(g_tpr);
                int res = getrf_caller<Real>::value(
                    LAPACK_COL_MAJOR, M, N, a, M, ipiv);
                handle_error(res);
            }

            template<typename Real>
            NO_INLINE void getri(Real* a, const lapack_int* ipiv, std::size_t N)
            {
                tp::sampler smp(g_tpr);
                Real workspace_query;
                int res = getri_caller<Real>::value(LAPACK_COL_MAJOR,
                    N, a, N, ipiv, &workspace_query, -1);
                handle_error(res);
                smp.do_sample();
                util::buffer<Real> work{ static_cast<std::uint64_t>(workspace_query) };
                res = getri_caller<Real>::value(
                    LAPACK_COL_MAJOR, N, a, N, ipiv, work.get(), work.size());
                handle_error(res);
            }

            template<typename Real>
            NO_INLINE void getrs(
                const Real* a,
                Real* b,
                const lapack_int* ipiv,
                std::size_t N,
                std::size_t Nrhs)
            {
                tp::sampler smp(g_tpr);
                int res = getrs_caller<Real>::value(
                    LAPACK_COL_MAJOR, 'N', N, Nrhs, a, N, ipiv, b, N);
                handle_error(res);
            }

            template<typename Real>
            NO_INLINE void tptri(Real* a, std::size_t N)
            {
                tp::sampler smp(g_tpr);
                int res = tptri_caller<Real>::value(
                    LAPACK_COL_MAJOR, 'L', 'N', N, a);
                handle_error(res);
            }

            template<typename Real>
            NO_INLINE void trtri(Real* a, std::size_t N)
            {
                tp::sampler smp(g_tpr);
                // upper triangular in row-major is lower triangular in column-major,
                // therefore pass 'L' to function which expects a column-major format
                int res = trtri_caller<Real>::value(
                    LAPACK_COL_MAJOR, 'L', 'N', N, a, N);
                handle_error(res);
            }

            template<typename Real>
            NO_INLINE void potrf(Real* a, std::size_t N)
            {
                tp::sampler smp(g_tpr);
                // upper triangular in row-major is lower triangular in column-major,
                // therefore pass 'L' to function which expects a column-major format
                int res = potrf_caller<Real>::value(
                    LAPACK_COL_MAJOR, 'L', N, a, N);
                handle_error(res);
            }
        }

        template<typename Real>
        void gesv_impl(std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            util::buffer<Real> a{ N * N };
            util::buffer<Real> b{ N * Nrhs };
            util::buffer<lapack_int> ipiv{ N };
            auto gen = [&]() { return dist(engine); };
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);
            compute::gesv(a.get(), b.get(), ipiv.get(), N, Nrhs);
        }

        template<typename Real>
        void gels_impl(std::size_t M, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            util::buffer<Real> a{ M * N };
            util::buffer<Real> b{ std::max(N, M) * Nrhs };
            auto gen = [&]() { return dist(engine); };
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);
            compute::gels(a.get(), b.get(), M, N, Nrhs);
        }

        template<typename Real>
        void getri_impl(std::size_t N, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            util::buffer<Real> a{ N * N };
            util::buffer<lapack_int> ipiv{ N };
            std::generate(a.begin(), a.end(), [&]() { return dist(engine); });
            compute::getrf(a.get(), ipiv.get(), N, N);
            compute::getri(a.get(), ipiv.get(), N);
        }

        template<typename Real>
        void getrf_impl(std::size_t M, std::size_t N, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            util::buffer<Real> a{ M * N };
            util::buffer<lapack_int> ipiv{ std::min(M, N) };
            auto gen = [&]() { return dist(engine); };
            std::generate(a.begin(), a.end(), gen);
            compute::getrf(a.get(), ipiv.get(), M, N);
        }

        template<typename Real>
        void getrs_impl(std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            util::buffer<Real> a{ N * N };
            util::buffer<Real> b{ N * Nrhs };
            util::buffer<lapack_int> ipiv{ N };
            auto gen = [&]() { return dist(engine); };
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);
            compute::getrf(a.get(), ipiv.get(), N, N);
            compute::getrs(a.get(), b.get(), ipiv.get(), N, Nrhs);
        }

        template<typename Real>
        void tptri_impl(std::size_t N, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            std::uniform_real_distribution<Real> dist{ 1.0, 2.0 };
            util::buffer<Real> a_packed{ N * (N + 1) / 2 };
            auto gen = [&]() { return dist(engine); };
            std::generate(a_packed.begin(), a_packed.end(), gen);
            compute::tptri(a_packed.get(), N);
        }

        template<typename Real>
        void trtri_impl(std::size_t N, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            std::uniform_real_distribution<Real> dist{ 1.0, 2.0 };
            util::buffer<Real> a{ N * N };
            auto gen = [&]() { return dist(engine); };
            fill_upper_triangular(a.begin(), a.end(), N, gen);
            compute::trtri(a.get(), N);
        }

        template<typename Real>
        void potrf_impl(std::size_t N, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            auto a = upper_dd_matrix<Real>(N, engine);
            compute::potrf(a.get(), N);
        }
    }

    NO_INLINE void dgesv(compute_params p, std::mt19937_64& engine)
    {
        detail::gesv_impl<double>(p.N, p.Nrhs, engine);
    }

    NO_INLINE void sgesv(compute_params p, std::mt19937_64& engine)
    {
        detail::gesv_impl<float>(p.N, p.Nrhs, engine);
    }

    NO_INLINE void dgetrs(compute_params p, std::mt19937_64& engine)
    {
        detail::getrs_impl<double>(p.N, p.Nrhs, engine);
    }

    NO_INLINE void sgetrs(compute_params p, std::mt19937_64& engine)
    {
        detail::getrs_impl<float>(p.N, p.Nrhs, engine);
    }

    NO_INLINE void dgels(compute_params p, std::mt19937_64& engine)
    {
        detail::gels_impl<double>(p.M, p.N, p.Nrhs, engine);
    }

    NO_INLINE void sgels(compute_params p, std::mt19937_64& engine)
    {
        detail::gels_impl<float>(p.M, p.N, p.Nrhs, engine);
    }

    NO_INLINE void dgetri(compute_params p, std::mt19937_64& engine)
    {
        detail::getri_impl<double>(p.N, engine);
    }

    NO_INLINE void sgetri(compute_params p, std::mt19937_64& engine)
    {
        detail::getri_impl<float>(p.N, engine);
    }

    NO_INLINE void dgetrf(compute_params p, std::mt19937_64& engine)
    {
        detail::getrf_impl<double>(p.M, p.N, engine);
    }

    NO_INLINE void sgetrf(compute_params p, std::mt19937_64& engine)
    {
        detail::getrf_impl<float>(p.M, p.N, engine);
    }

    NO_INLINE void dtptri(compute_params p, std::mt19937_64& engine)
    {
        detail::tptri_impl<double>(p.N, engine);
    }

    NO_INLINE void stptri(compute_params p, std::mt19937_64& engine)
    {
        detail::tptri_impl<float>(p.N, engine);
    }

    NO_INLINE void dtrtri(compute_params p, std::mt19937_64& engine)
    {
        detail::trtri_impl<double>(p.N, engine);
    }

    NO_INLINE void strtri(compute_params p, std::mt19937_64& engine)
    {
        detail::trtri_impl<float>(p.N, engine);
    }

    NO_INLINE void dpotrf(compute_params p, std::mt19937_64& engine)
    {
        detail::potrf_impl<double>(p.N, engine);
    }

    NO_INLINE void spotrf(compute_params p, std::mt19937_64& engine)
    {
        detail::potrf_impl<float>(p.N, engine);
    }

    class cmdparams
    {
        using work_func = decltype(&dgesv);
        work_func func = nullptr;
        compute_params params = {};

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
                    too_few(argv[0], std::move(op_type));
                util::to_scalar(argv[2], params.M);
                assert_positive(params.M, "m");
                util::to_scalar(argv[3], params.N);
                assert_positive(params.N, "n");
                util::to_scalar(argv[4], params.Nrhs);
                assert_positive(params.Nrhs, "nrhs");
                get_iterations(argc, argv, 5);
            }
            else if (func == dgesv || func == sgesv || func == dgetrs || func == sgetrs)
            {
                if (argc < 4)
                    too_few(argv[0], std::move(op_type));
                util::to_scalar(argv[2], params.N);
                assert_positive(params.N, "n");
                util::to_scalar(argv[3], params.Nrhs);
                assert_positive(params.Nrhs, "nrhs");
                get_iterations(argc, argv, 4);
            }
            else if (single_arg(func))
            {
                if (argc < 3)
                    too_few(argv[0], std::move(op_type));
                util::to_scalar(argv[2], params.N);
                assert_positive(params.N, "n");
                get_iterations(argc, argv, 3);
            }
            else if (func == dgetrf || func == sgetrf)
            {
                if (argc < 4)
                    too_few(argv[0], std::move(op_type));
                util::to_scalar(argv[2], params.M);
                assert_positive(params.M, "m");
                util::to_scalar(argv[3], params.N);
                assert_positive(params.N, "n");
                get_iterations(argc, argv, 4);
            }
        }

        void do_work(std::mt19937_64& engine) const
        {
            func(params, engine);
        }

    private:
        bool is_inversion(work_func func)
        {
            return func == dgetri || func == sgetri ||
                func == dtptri || func == stptri ||
                func == dtrtri || func == strtri;
        }

        bool single_arg(work_func func)
        {
            return is_inversion(func) || func == dpotrf || func == spotrf;
        }

        void assert_positive(std::size_t x, std::string name)
        {
            assert(x);
            if (!x)
                throw std::invalid_argument(std::move(name.append(" must be greater than 0")));
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
            if (str == "dgetrs")
                return dgetrs;
            if (str == "sgetrs")
                return sgetrs;
            if (str == "dpotrf")
                return dpotrf;
            if (str == "spotrf")
                return spotrf;
            throw std::invalid_argument(std::string("invalid work type: ")
                .append(str));
        }

        void too_few(const char* prog, std::string op)
        {
            print_usage(prog);
            throw std::invalid_argument(
                std::move(op.append(": too few arguments")));
        }

    #if defined(USE_ITERATIONS)
        void print_usage(const char* prog)
        {
            std::cerr << "Usage:\n"
                << "\t" << prog << " {dgesv,sgesv,dgetrs,sgetrs} <n> <nrhs> <iters>\n"
                << "\t" << prog << " {dgels,sgels} <m> <n> <nrhs> <iters>\n"
                << "\t" << prog << " {dgetri,sgetri} <n> <iters>\n"
                << "\t" << prog << " {dtptri,stptri,dtrtri,strtri} <n> <iters>\n"
                << "\t" << prog << " {dpotrf,spotrf} <n> <iters>\n"
                << "\t" << prog << " {dgetrf,sgetrf} <m> <n> <iters>\n";
        }

        void get_iterations(int argc, const char* const* argv, int idx)
        {
            if (argc > idx)
                util::to_scalar(argv[idx], params.iters);
            assert_positive(params.iters, "iters");
        }
    #else // !defined(USE_ITERATIONS)
        void print_usage(const char* prog)
        {
            std::cerr << "Usage:\n"
                << "\t" << prog << " {dgesv,sgesv,dgetrs,sgetrs} <n> <nrhs>\n"
                << "\t" << prog << " {dgels,sgels} <m> <n> <nrhs>\n"
                << "\t" << prog << " {dgetri,sgetri} <n>\n"
                << "\t" << prog << " {dtptri,stptri,dtrtri,strtri} <n>\n"
                << "\t" << prog << " {dpotrf,spotrf} <n>\n"
                << "\t" << prog << " {dgetrf,sgetrf} <m> <n>\n";
        }

        void get_iterations(int, const char* const*, int) {}
    #endif // defined(USE_ITERATIONS)
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
        return 1;
    }
}
