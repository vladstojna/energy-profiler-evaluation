#include <timeprinter/printer.hpp>
#include <util/cuda_utils.hpp>
#include <util/to_scalar.hpp>
#include <util/unique_handle.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cassert>
#include <random>

namespace
{
    tp::printer g_tpr;

    template<auto Func>
    struct func_obj : std::integral_constant<decltype(Func), Func> {};

    cusolverDnHandle_t cusolverdn_create()
    {
        cusolverDnHandle_t handle;
        auto status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("Error creating cuSOLVER");
        return handle;
    }

    void cusolverdn_destroy(cusolverDnHandle_t handle)
    {
        auto status = cusolverDnDestroy(handle);
        if (status != CUSOLVER_STATUS_SUCCESS)
            std::cerr << "Error destroying cuSOLVER";
    }

    using cusolverdn_handle =
        util::unique_handle<cusolverDnHandle_t, func_obj<cusolverdn_destroy>>;

    namespace detail
    {
    #define DEFINE_CALL_MEMBERS(prefix, prec) \
        static constexpr auto query = cusolverDn ## prec ## prefix ## _bufferSize; \
        static constexpr const char query_str[] = "cusolverDn" #prec #prefix "_bufferSize"; \
        static constexpr auto compute = cusolverDn ## prec ## prefix; \
        static constexpr const char compute_str[] = "cusolverDn" #prec #prefix

    #define DEFINE_CALL_ANY(prefix, prec_single, prec_double) \
        template<typename> \
        struct prefix ## _call {}; \
        template<> \
        struct prefix ## _call<float> \
        { \
            DEFINE_CALL_MEMBERS(prefix, prec_single); \
        }; \
        template<> \
        struct prefix ## _call<double> \
        { \
            DEFINE_CALL_MEMBERS(prefix, prec_double); \
        }

    #define DEFINE_CALL_IRS(prefix) DEFINE_CALL_ANY(prefix, SS, DD)
    #define DEFINE_CALL(prefix) DEFINE_CALL_ANY(prefix, S, D)

    #if CUDART_VERSION >= 10020
        DEFINE_CALL_IRS(gesv);
    #endif // CUDART_VERSION >= 10020

    #if CUDART_VERSION >= 11000
        DEFINE_CALL_IRS(gels);
    #endif // CUDART_VERSION >= 11000

    #if CUDART_VERSION < 11010
        using index_type = cusolver_int_t;
    #else
        using index_type = std::int64_t;
    #endif // CUDART_VERSION < 11010

        template<typename>
        struct cuda_data_type {};

        template<>
        struct cuda_data_type<float> :
            std::integral_constant<decltype(CUDA_R_32F), CUDA_R_32F>
        {};

        template<>
        struct cuda_data_type<double> :
            std::integral_constant<decltype(CUDA_R_64F), CUDA_R_64F>
        {};

        void cusolver_error(const char* func, cusolverStatus_t status)
        {
            throw std::runtime_error(std::string(func)
                .append(" error, status = ").append(std::to_string(status)));
        }

    #if CUDART_VERSION < 11040
        std::pair<int, int> cudart_separate_version(int version)
        {
            return { version / 1000, (version % 1000) / 10 };
        }

        std::string unsupported_version(std::string_view feature, std::string_view required)
        {
            auto [major, minor] = cudart_separate_version(CUDART_VERSION);
            return std::string(feature)
                .append(" requires CUDA Toolkit v")
                .append(required)
                .append(" or higher, found CUDA runtime v")
                .append(std::to_string(major))
                .append(".")
                .append(std::to_string(minor));
        }
    #endif // CUDART_VERSION < 11040

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

            util::buffer<Real> a(N * N);
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

    #if CUDART_VERSION < 11040

        DEFINE_CALL(trtri);

        template<typename Real>
        int trtri_compute(cusolverdn_handle& handle, std::size_t N, util::device_buffer<Real>& a)
        {
            using call = trtri_call<Real>;

            constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
            constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

            tp::sampler smp(g_tpr);

            int info = 0;
            util::device_buffer dev_info{ &info, &info + 1 };
            int lwork;
            auto status = call::query(handle, uplo, diag, N, a.get(), N, &lwork);
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::query_str, status);

            smp.do_sample();

            util::device_buffer<Real> dev_work{ static_cast<std::size_t>(lwork) };

            status = call::compute(
                handle, uplo, diag, N, a.get(), N, dev_work.get(), lwork, dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::compute_str, status);

            util::copy(dev_info, 1, &info);
            return info;
        }
    #else
        template<typename Real>
        int trtri_compute(cusolverdn_handle& handle, std::size_t N, util::device_buffer<Real>& a)
        {
            constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
            constexpr cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

            tp::sampler smp(g_tpr);

            int info = 0;
            util::device_buffer dev_info{ &info, &info + 1 };

            std::size_t workspace_device;
            std::size_t workspace_host;
            auto status = cusolverDnXtrtri_bufferSize(handle, uplo, diag, N,
                cuda_data_type<Real>::value,
                a.get(), N,
                &workspace_device, &workspace_host);
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error("cusolverDnXtrtri_bufferSize", status);

            smp.do_sample();

            util::device_buffer<std::uint8_t> dev_work{ workspace_device };
            util::buffer<std::uint8_t> host_work;
            if (workspace_host)
                host_work = util::buffer<std::uint8_t>{ workspace_host };

            status = cusolverDnXtrtri(handle, uplo, diag, N,
                cuda_data_type<Real>::value,
                a.get(), N,
                dev_work.get(), workspace_device,
                host_work.get(), workspace_host,
                dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error("cusolverDnXtrtri", status);

            util::copy(dev_info, 1, &info);
            return info;
        }
    #endif // CUDART_VERSION < 11040

        template<typename Real>
        int trtri_impl(cusolverdn_handle& handle, std::size_t N, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 1.0, 2.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);

            util::buffer<Real> a{ N * N };
            std::fill(a.begin(), a.end(), Real{});
            fill_upper_triangular(a.begin(), a.end(), N, gen);

            smp.do_sample();

            util::device_buffer dev_a{ a.begin(), a.end() };
            // upper triangular in row-major is lower triangular in column-major,
            // therefore pass 'L' to function which expects a column-major format
            int info = trtri_compute(handle, N, dev_a);

            util::copy(dev_a, dev_a.size(), a.begin());
            return info;
        }

    #if CUDART_VERSION < 11010

        DEFINE_CALL(getrf);

        template<typename Real>
        int getrf_compute(
            cusolverdn_handle& handle,
            std::size_t M,
            std::size_t N,
            util::device_buffer<Real>& a,
            util::device_buffer<index_type>& ipiv)
        {
            using call = getrf_call<Real>;
            tp::sampler smp(g_tpr);

            int info = 0;
            util::device_buffer dev_info{ &info, &info + 1 };
            int lwork;
            auto status = call::query(handle, M, N, a.get(), M, &lwork);
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::query_str, status);

            smp.do_sample();

            util::device_buffer<Real> dev_work{ static_cast<std::size_t>(lwork) };

            status = call::compute(
                handle, M, N, a.get(), M, dev_work.get(), ipiv.get(), dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::compute_str, status);

            util::copy(dev_info, 1, &info);
            return info;
        }
    #else
        template<typename Real>
        int getrf_compute(
            cusolverdn_handle& handle,
            std::size_t M,
            std::size_t N,
            util::device_buffer<Real>& a,
            util::device_buffer<index_type>& ipiv)
        {
            tp::sampler smp(g_tpr);

            int info = 0;
            util::device_buffer dev_info{ &info, &info + 1 };
            std::size_t workspace_device;
            std::size_t workspace_host;
            auto status = cusolverDnXgetrf_bufferSize(handle, nullptr, M, N,
                cuda_data_type<Real>::value,
                a.get(), M,
                cuda_data_type<Real>::value,
                &workspace_device,
                &workspace_host);
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error("cusolverDnXgetrf_bufferSize", status);

            smp.do_sample();

            util::device_buffer<std::uint8_t> dev_work{ workspace_device };
            util::buffer<std::uint8_t> host_work;
            if (workspace_host)
                host_work = util::buffer<std::uint8_t>{ workspace_host };

            status = cusolverDnXgetrf(handle, nullptr, M, N,
                cuda_data_type<Real>::value,
                a.get(), M,
                ipiv.get(),
                cuda_data_type<Real>::value,
                dev_work.get(), workspace_device,
                host_work.get(), workspace_host,
                dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error("cusolverDnXgetrf", status);

            util::copy(dev_info, 1, &info);
            return info;
        }
    #endif // CUDART_VERSION < 11010

    #if CUDART_VERSION < 11010
        template<typename>
        struct getrs_call {};
        template<>
        struct getrs_call<float>
        {
            static constexpr auto compute = cusolverDnSgetrs;
            static constexpr const char compute_str[] = "cusolverDnSgetrs";
        };
        template<>
        struct getrs_call<double>
        {
            static constexpr auto compute = cusolverDnDgetrs;
            static constexpr const char compute_str[] = "cusolverDnDgetrs";
        };

        template<typename Real>
        int getrs_compute(
            cusolverdn_handle& handle,
            std::size_t N,
            std::size_t Nrhs,
            const util::device_buffer<Real>& a,
            const util::device_buffer<index_type>& ipiv,
            util::device_buffer<Real>& b)
        {
            constexpr cublasOperation_t op = CUBLAS_OP_N;
            tp::sampler smp(g_tpr);

            int info = 0;
            util::device_buffer dev_info{ &info, &info + 1 };

            auto status = getrs_call<Real>::compute(
                handle, op, N, Nrhs, a.get(), N, ipiv.get(), b.get(), N, dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(getrs_call<Real>::compute_str, status);

            util::copy(dev_info, 1, &info);
            return info;
        }
    #else
        template<typename Real>
        int getrs_compute(
            cusolverdn_handle& handle,
            std::size_t N,
            std::size_t Nrhs,
            const util::device_buffer<Real>& a,
            const util::device_buffer<index_type>& ipiv,
            util::device_buffer<Real>& b)
        {
            constexpr cublasOperation_t op = CUBLAS_OP_N;
            tp::sampler smp(g_tpr);

            int info = 0;
            util::device_buffer dev_info{ &info, &info + 1 };

            auto status = cusolverDnXgetrs(handle, nullptr, op, N, Nrhs,
                cuda_data_type<Real>::value,
                a.get(), N,
                ipiv.get(),
                cuda_data_type<Real>::value,
                b.get(), N,
                dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error("cusolverDnXgetrs", status);

            util::copy(dev_info, 1, &info);
            return info;
        }
    #endif // CUDART_VERSION < 11010

        template<typename Real>
        int getrf_impl(
            cusolverdn_handle& handle, std::size_t M, std::size_t N, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);

            util::buffer<Real> a{ M * N };
            util::buffer<index_type> ipiv{ std::min(M, N) };
            std::generate(a.begin(), a.end(), gen);

            smp.do_sample();

            util::device_buffer dev_a{ a.begin(), a.end() };
            util::device_buffer<index_type> dev_ipiv{ ipiv.size() };

            int info = getrf_compute(handle, M, N, dev_a, dev_ipiv);

            util::copy(dev_a, dev_a.size(), a.begin());
            util::copy(dev_ipiv, dev_ipiv.size(), ipiv.begin());
            return info;
        }

        template<typename Real>
        int getrs_impl(
            cusolverdn_handle& handle, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);

            util::buffer<Real> a{ N * N };
            util::buffer<Real> b{ N * Nrhs };
            util::buffer<index_type> ipiv{ N };
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            smp.do_sample();

            util::device_buffer dev_a{ a.begin(), a.end() };
            util::device_buffer dev_b{ b.begin(), b.end() };
            util::device_buffer<index_type> dev_ipiv{ ipiv.size() };

            int info = getrf_compute(handle, N, N, dev_a, dev_ipiv);
            if (info)
            {
                std::cerr << "getrs_impl: getrf_compute error\n";
                return info;
            }

            info = getrs_compute(handle, N, Nrhs, dev_a, dev_ipiv, dev_b);

            util::copy(dev_b, dev_b.size(), b.begin());
            return info;
        }

    #if CUDART_VERSION < 10020
        template<typename Real>
        int gesv_impl(cusolverdn_handle&, std::size_t, std::size_t, std::mt19937_64&)
        {
            throw std::runtime_error(unsupported_version("cusolverDn<t1><t2>gesv()", "10.2"));
        }
    #else
        template<typename Real>
        int gesv_impl(
            cusolverdn_handle& handle, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
        {
            using call = gesv_call<Real>;

            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);

            util::buffer<Real> a{ N * N };
            util::buffer<Real> b{ N * Nrhs };
            util::buffer<Real> x{ N * Nrhs };
            util::buffer<cusolver_int_t> ipiv{ N };
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            smp.do_sample();

            util::device_buffer dev_a{ a.begin(), a.end() };
            util::device_buffer dev_b{ b.begin(), b.end() };
            util::device_buffer<Real> dev_x{ x.size() };
            util::device_buffer<cusolver_int_t> dev_ipiv{ ipiv.size() };

            smp.do_sample();

            std::size_t work_bytes;
            auto status = call::query(handle, N, Nrhs,
                nullptr, N, nullptr, nullptr, N, nullptr, N, nullptr, &work_bytes);
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::query_str, status);

            smp.do_sample();

            cusolver_int_t iters;
            cusolver_int_t info = 0;
            util::device_buffer<cusolver_int_t> dev_info{ &info, &info + 1 };
            util::device_buffer<std::uint8_t> dev_work{ work_bytes };

            status = call::compute(handle, N, Nrhs,
                dev_a.get(), N,
                dev_ipiv.get(),
                dev_b.get(), N,
                dev_x.get(), N,
                dev_work.get(), work_bytes,
                &iters,
                dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::compute_str, status);
            cudaDeviceSynchronize();

            smp.do_sample();

            std::cerr << "iterations = " << iters << "\n";
            util::copy(dev_info, 1, &info);
            util::copy(dev_x, dev_x.size(), x.begin());
            util::copy(dev_ipiv, dev_ipiv.size(), ipiv.begin());
            util::copy(dev_a, dev_a.size(), a.begin());
            return info;
        }
    #endif // CUDART_VERSION < 10020

    #if CUDART_VERSION < 11000
        template<typename Real>
        int gels_impl(cusolverdn_handle&, std::size_t, std::size_t, std::size_t, std::mt19937_64&)
        {
            throw std::runtime_error(unsupported_version("cusolverDn<t1><t2>gels()", "11.0"));
        }
    #else
        template<typename Real>
        int gels_impl(
            cusolverdn_handle& handle, std::size_t M, std::size_t N,
            std::size_t Nrhs, std::mt19937_64& engine)
        {
            using call = gels_call<Real>;

            assert(N <= M);
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);

            util::buffer<Real> a{ M * N };
            util::buffer<Real> b{ M * Nrhs };
            util::buffer<Real> x{ N * Nrhs };
            std::generate(a.begin(), a.end(), gen);
            std::generate(b.begin(), b.end(), gen);

            smp.do_sample();

            util::device_buffer dev_a{ a.begin(), a.end() };
            util::device_buffer dev_b{ b.begin(), b.end() };
            util::device_buffer<Real> dev_x{ x.size() };

            smp.do_sample();

            std::size_t work_bytes;
            auto status = call::query(handle, M, N, Nrhs,
                nullptr, M, nullptr, M, nullptr, N, nullptr, &work_bytes);
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::query_str, status);

            smp.do_sample();

            cusolver_int_t iters;
            cusolver_int_t info = 0;
            util::device_buffer<cusolver_int_t> dev_info{ &info, &info + 1 };
            util::device_buffer<std::uint8_t> dev_work{ work_bytes };

            status = call::compute(handle, M, N, Nrhs,
                dev_a.get(), M,
                dev_b.get(), M,
                dev_x.get(), N,
                dev_work.get(), work_bytes,
                &iters,
                dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::compute_str, status);
            cudaDeviceSynchronize();

            smp.do_sample();

            std::cerr << "iterations = " << iters << "\n";
            util::copy(dev_info, 1, &info);
            util::copy(dev_x, dev_x.size(), x.begin());
            util::copy(dev_a, dev_a.size(), a.begin());
            return info;
        }
    #endif // CUDART_VERSION < 11000

    #if CUDART_VERSION >= 11010
        // cusolverDnXpotrf()
        template<typename Real>
        int potrf_compute(cusolverdn_handle& handle, std::size_t N, util::device_buffer<Real>& a)
        {
            constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

            tp::sampler smp(g_tpr);

            int info = 0;
            util::device_buffer dev_info{ &info, &info + 1 };

            std::size_t workspace_device;
            std::size_t workspace_host;
            auto status = cusolverDnXpotrf_bufferSize(handle, nullptr, uplo, N,
                cuda_data_type<Real>::value,
                a.get(), N,
                cuda_data_type<Real>::value,
                &workspace_device,
                &workspace_host);
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error("cusolverDnXpotrf_bufferSize", status);

            smp.do_sample();

            util::device_buffer<std::uint8_t> dev_work{ workspace_device };
            util::buffer<std::uint8_t> host_work;
            if (workspace_host)
                host_work = util::buffer<std::uint8_t>{ workspace_host };

            status = cusolverDnXpotrf(handle, nullptr, uplo, N,
                cuda_data_type<Real>::value,
                a.get(), N,
                cuda_data_type<Real>::value,
                dev_work.get(), workspace_device,
                host_work.get(), workspace_host,
                dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error("cusolverDnXpotrf", status);

            util::copy(dev_info, 1, &info);
            return info;
        }
    #elif CUDART_VERSION >= 11000
        // cusolverDnPotrf()
        template<typename Real>
        int potrf_compute(cusolverdn_handle& handle, std::size_t N, util::device_buffer<Real>& a)
        {
            constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

            tp::sampler smp(g_tpr);

            int info = 0;
            util::device_buffer dev_info{ &info, &info + 1 };

            std::size_t workspace;
            auto status = cusolverDnPotrf_bufferSize(handle, nullptr, uplo, N,
                cuda_data_type<Real>::value,
                a.get(), N,
                cuda_data_type<Real>::value,
                &workspace);
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error("cusolverDnPotrf_bufferSize", status);

            smp.do_sample();

            util::device_buffer<std::uint8_t> dev_work{ workspace };
            status = cusolverDnPotrf(handle, nullptr, uplo, N,
                cuda_data_type<Real>::value,
                a.get(), N,
                cuda_data_type<Real>::value,
                dev_work.get(), workspace,
                dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error("cusolverDnPotrf", status);

            util::copy(dev_info, 1, &info);
            return info;
        }
    #else
        DEFINE_CALL(potrf);

        // cusolverDn<t>potrf()
        template<typename Real>
        int potrf_compute(cusolverdn_handle& handle, std::size_t N, util::device_buffer<Real>& a)
        {
            using call = potrf_call<Real>;

            constexpr cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

            tp::sampler smp(g_tpr);

            int info = 0;
            util::device_buffer dev_info{ &info, &info + 1 };
            int lwork;
            auto status = call::query(handle, uplo, N, a.get(), N, &lwork);
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::query_str, status);

            smp.do_sample();

            util::device_buffer<Real> dev_work{ static_cast<std::size_t>(lwork) };

            status = call::compute(
                handle, uplo, N, a.get(), N, dev_work.get(), lwork, dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                cusolver_error(call::compute_str, status);

            util::copy(dev_info, 1, &info);
            return info;
        }
    #endif // CUDART_VERSION >= 11010

        template<typename Real>
        int potrf_impl(cusolverdn_handle& handle, std::size_t N, std::mt19937_64& engine)
        {
            tp::sampler smp(g_tpr);
            util::buffer<Real> a = upper_dd_matrix<Real>(N, engine);
            util::device_buffer dev_a{ a.begin(), a.end() };
            // upper triangular in row-major is lower triangular in column-major,
            // therefore pass 'L' to function which expects a column-major format
            int info = potrf_compute(handle, N, dev_a);
            util::copy(dev_a, dev_a.size(), a.begin());
            return info;
        }
    }

    __attribute__((noinline))
        int dtrtri(cusolverdn_handle& handle, std::size_t N, std::mt19937_64& engine)
    {
        return detail::trtri_impl<double>(handle, N, engine);
    }

    __attribute__((noinline))
        int strtri(cusolverdn_handle& handle, std::size_t N, std::mt19937_64& engine)
    {
        return detail::trtri_impl<float>(handle, N, engine);
    }

    __attribute__((noinline))
        int dgetrf(cusolverdn_handle& handle, std::size_t M, std::size_t N, std::mt19937_64& engine)
    {
        return detail::getrf_impl<double>(handle, M, N, engine);
    }

    __attribute__((noinline))
        int sgetrf(cusolverdn_handle& handle, std::size_t M, std::size_t N, std::mt19937_64& engine)
    {
        return detail::getrf_impl<float>(handle, M, N, engine);
    }

    __attribute__((noinline))
        int dgetrs(
            cusolverdn_handle& handle, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
    {
        return detail::getrs_impl<double>(handle, N, Nrhs, engine);
    }

    __attribute__((noinline))
        int sgetrs(
            cusolverdn_handle& handle, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
    {
        return detail::getrs_impl<float>(handle, N, Nrhs, engine);
    }

    __attribute__((noinline))
        int dgesv(
            cusolverdn_handle& handle, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
    {
        return detail::gesv_impl<double>(handle, N, Nrhs, engine);
    }

    __attribute__((noinline))
        int sgesv(
            cusolverdn_handle& handle, std::size_t N, std::size_t Nrhs, std::mt19937_64& engine)
    {
        return detail::gesv_impl<float>(handle, N, Nrhs, engine);
    }

    __attribute__((noinline))
        int dgels(
            cusolverdn_handle& handle, std::size_t M, std::size_t N,
            std::size_t Nrhs, std::mt19937_64& engine)
    {
        return detail::gels_impl<double>(handle, M, N, Nrhs, engine);
    }

    __attribute__((noinline))
        int sgels(
            cusolverdn_handle& handle, std::size_t M, std::size_t N,
            std::size_t Nrhs, std::mt19937_64& engine)
    {
        return detail::gels_impl<float>(handle, M, N, Nrhs, engine);
    }

    __attribute__((noinline))
        int dpotrf(cusolverdn_handle& handle, std::size_t N, std::mt19937_64& engine)
    {
        return detail::potrf_impl<double>(handle, N, engine);
    }

    __attribute__((noinline))
        int spotrf(cusolverdn_handle& handle, std::size_t N, std::mt19937_64& engine)
    {
        return detail::potrf_impl<float>(handle, N, engine);
    }

    namespace work_type
    {
        enum type
        {
            dtrtri,
            strtri,
            dgetrf,
            sgetrf,
            dgetrs,
            sgetrs,
            dgesv,
            sgesv,
            dgels,
            sgels,
            dpotrf,
            spotrf,
        };
    }

    struct cmdargs
    {
        std::size_t m = 0;
        std::size_t n = 0;
        std::size_t nrhs = 0;
        work_type::type wtype = static_cast<work_type::type>(0);

        cmdargs(int argc, const char* const* argv)
        {
            const char* prog = argv[0];
            if (argc < 2)
            {
                usage(prog);
                throw std::invalid_argument("Not enough arguments");
            }
            std::string op_type(argv[1]);
            std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                [](unsigned char c) { return std::tolower(c); });
            wtype = get_work_type(op_type);
            assert(wtype);

            if (wtype == work_type::dgetrf || wtype == work_type::sgetrf)
            {
                if (argc < 4)
                    throw_too_few(prog, op_type);
                util::to_scalar(argv[2], m);
                check_positive(m, "m");
                util::to_scalar(argv[3], n);
                check_positive(n, "n");
            }
            else if (wtype == work_type::dgetrs || wtype == work_type::sgetrs ||
                wtype == work_type::dgesv || wtype == work_type::sgesv)
            {
                if (argc < 4)
                    throw_too_few(prog, op_type);
                util::to_scalar(argv[2], n);
                check_positive(n, "n");
                util::to_scalar(argv[3], nrhs);
                check_positive(nrhs, "n_rhs");
            }
            else if (wtype == work_type::dgels || wtype == work_type::sgels)
            {
                if (argc < 5)
                    throw_too_few(prog, op_type);
                util::to_scalar(argv[2], m);
                check_positive(m, "m");
                util::to_scalar(argv[3], n);
                check_positive(n, "n");
                util::to_scalar(argv[4], nrhs);
                check_positive(nrhs, "n_rhs");
                if (n > m)
                    throw std::invalid_argument("n must not be greater than m");
            }
            else if (wtype == work_type::dtrtri || wtype == work_type::strtri ||
                wtype == work_type::dpotrf || wtype == work_type::spotrf)
            {
                if (argc < 3)
                    throw_too_few(prog, op_type);
                util::to_scalar(argv[2], n);
                check_positive(n, "n");
            }
        }

    private:
        static void check_positive(std::size_t val, const char* val_str)
        {
            if (!val)
                throw std::invalid_argument(std::string(val_str)
                    .append(" must be a positive integer"));
            assert(val);
        }

        static void throw_too_few(const char* prog, std::string& op_type)
        {
            usage(prog);
            throw std::invalid_argument(op_type.append(": too few arguments"));
        }

        static work_type::type get_work_type(const std::string& op_type)
        {
        #define WORK_RETURN_IF(op_type, arg) \
            if (op_type == #arg) \
                return work_type::arg

            WORK_RETURN_IF(op_type, dtrtri);
            WORK_RETURN_IF(op_type, strtri);
            WORK_RETURN_IF(op_type, dgetrf);
            WORK_RETURN_IF(op_type, sgetrf);
            WORK_RETURN_IF(op_type, dgetrs);
            WORK_RETURN_IF(op_type, sgetrs);
            WORK_RETURN_IF(op_type, dgesv);
            WORK_RETURN_IF(op_type, sgesv);
            WORK_RETURN_IF(op_type, dgels);
            WORK_RETURN_IF(op_type, sgels);
            WORK_RETURN_IF(op_type, dpotrf);
            WORK_RETURN_IF(op_type, spotrf);
            throw std::invalid_argument(std::string("Unknown work type ").append(op_type));
        }

        static void usage(const char* prog)
        {
            std::cerr << "Usage:\n"
                << "\t" << prog << " {dtrtri,strtri} <n>\n"
                << "\t" << prog << " {dpotrf,spotrf} <n>\n"
                << "\t" << prog << " {dgetrf,sgetrf} <m> <n>\n"
                << "\t" << prog << " {dgetrs,sgetrs,dgesv,sgesv} <n> <n_rhs>\n"
                << "\t" << prog << " {dgels,sgels} <n> <m> <n_rhs>\n";
        }
    };

    int execute_work(const cmdargs& args, cusolverdn_handle& handle, std::mt19937_64& gen)
    {
        switch (args.wtype)
        {
        case work_type::dtrtri:
            std::cerr << "dtrtri N=" << args.n << "\n";
            return dtrtri(handle, args.n, gen);
        case work_type::strtri:
            std::cerr << "strtri N=" << args.n << "\n";
            return strtri(handle, args.n, gen);
        case work_type::dpotrf:
            std::cerr << "dpotrf N=" << args.n << "\n";
            return dpotrf(handle, args.n, gen);
        case work_type::spotrf:
            std::cerr << "spotrf N=" << args.n << "\n";
            return spotrf(handle, args.n, gen);
        case work_type::dgetrf:
            std::cerr << "dgetrf M=" << args.m << ", N=" << args.n << "\n";
            return dgetrf(handle, args.m, args.n, gen);
        case work_type::sgetrf:
            std::cerr << "sgetrf M=" << args.m << ", N=" << args.n << "\n";
            return sgetrf(handle, args.m, args.n, gen);
        case work_type::dgetrs:
            std::cerr << "dgetrs N=" << args.n << ", Nrhs=" << args.nrhs << "\n";
            return dgetrs(handle, args.n, args.nrhs, gen);
        case work_type::sgetrs:
            std::cerr << "sgetrs N=" << args.n << ", Nrhs=" << args.nrhs << "\n";
            return sgetrs(handle, args.n, args.nrhs, gen);
        case work_type::dgesv:
            std::cerr << "dgesv N=" << args.n << ", Nrhs=" << args.nrhs << "\n";
            return dgesv(handle, args.n, args.nrhs, gen);
        case work_type::sgesv:
            std::cerr << "sgesv N=" << args.n << ", Nrhs=" << args.nrhs << "\n";
            return sgesv(handle, args.n, args.nrhs, gen);
        case work_type::dgels:
            std::cerr << "dgels M=" << args.m << ", N=" << args.n << ", Nrhs=" << args.nrhs << "\n";
            return dgels(handle, args.m, args.n, args.nrhs, gen);
        case work_type::sgels:
            std::cerr << "sgels M=" << args.m << ", N=" << args.n << ", Nrhs=" << args.nrhs << "\n";
            return sgels(handle, args.m, args.n, args.nrhs, gen);
        }
        throw std::runtime_error("Invalid work type");
    }

    void handle_info(int info)
    {
        if (info == 0)
            std::cerr << "Success, info = " << info << "\n";
        else if (info > 0)
            std::cerr << "Solution not found, info = " << info << "\n";
        else
            std::cerr << "Invalid parameter, info = " << info << "\n";
    }
}

int main(int argc, char** argv)
{
    try
    {
        const cmdargs args(argc, argv);
        std::random_device rnd_dev;
        std::mt19937_64 engine{ rnd_dev() };
        cusolverdn_handle handle{ cusolverdn_create() };
        handle_info(execute_work(args, handle, engine));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
