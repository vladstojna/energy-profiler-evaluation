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

        template<typename Real>
        int trtri_impl(cusolverdn_handle& handle, std::size_t N, std::mt19937_64& engine)
        {
            static constexpr const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
            static constexpr const cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

            static auto fill_lower_triangular = [](auto from, auto to, std::size_t ld, auto gen)
            {
                for (auto [it, non_zero] = std::pair{ from, 1 }; it < to; it += ld, non_zero++)
                    for (auto entry = it; entry < it + non_zero; entry++)
                        *entry = gen();
            };

            std::uniform_real_distribution<Real> dist{ 1.0, 2.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);

            int info = 0;
            util::buffer<Real> a{ N * N };
            std::fill(a.begin(), a.end(), Real{});
            fill_lower_triangular(a.begin(), a.end(), N, gen);

            smp.do_sample();

            util::device_buffer dev_a{ a.begin(), a.end() };
            util::device_buffer<int> dev_info{ 1 };
            if (auto status = cudaMemset(dev_info.get(), 0, sizeof(decltype(dev_info)::value_type));
                status != cudaSuccess)
            {
                throw util::device_exception(util::get_cuda_error_str("cudaMemset", status));
            }

            smp.do_sample();

            std::size_t workspace_device;
            std::size_t workspace_host;
            auto status = cusolverDnXtrtri_bufferSize(handle, uplo, diag, N,
                cuda_data_type<Real>::value,
                dev_a.get(), N,
                &workspace_device, &workspace_host);
            if (status != CUSOLVER_STATUS_SUCCESS)
                throw std::runtime_error("cusolverDnXtrtri_bufferSize error");

            smp.do_sample();

            util::device_buffer<std::uint8_t> dev_work{ workspace_device };
            util::buffer<std::uint8_t> host_work;
            if (workspace_host)
                host_work = util::buffer<std::uint8_t>{ workspace_host };

            status = cusolverDnXtrtri(handle, uplo, diag, N,
                cuda_data_type<Real>::value,
                dev_a.get(), N,
                dev_work.get(), workspace_device,
                host_work.get(), workspace_host,
                dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                throw std::runtime_error("cusolverDnXtrtri_bufferSize error");
            cudaDeviceSynchronize();

            smp.do_sample();

            util::copy(dev_a, a.begin(), a.size());
            util::copy(dev_info, &info, 1);
            return info;
        }

        template<typename Real>
        int getrf_impl(
            cusolverdn_handle& handle, std::size_t M, std::size_t N, std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            tp::sampler smp(g_tpr);

            int info = 0;
            util::buffer<Real> a{ M * N };
            util::buffer<std::int64_t> ipiv{ std::min(M, N) };
            std::generate(a.begin(), a.end(), gen);

            smp.do_sample();

            util::device_buffer dev_a{ a.begin(), a.end() };
            util::device_buffer<std::int64_t> dev_ipiv{ ipiv.size() };
            util::device_buffer<int> dev_info{ 1 };
            if (auto status = cudaMemset(dev_info.get(), 0, sizeof(decltype(dev_info)::value_type));
                status != cudaSuccess)
            {
                throw util::device_exception(util::get_cuda_error_str("cudaMemset", status));
            }

            smp.do_sample();

            std::size_t workspace_device;
            std::size_t workspace_host;
            auto status = cusolverDnXgetrf_bufferSize(handle, nullptr, M, N,
                cuda_data_type<Real>::value,
                dev_a.get(), M,
                cuda_data_type<Real>::value,
                &workspace_device,
                &workspace_host);
            if (status != CUSOLVER_STATUS_SUCCESS)
                throw std::runtime_error("cusolverDnXgetrf_bufferSize error");

            smp.do_sample();

            util::device_buffer<std::uint8_t> dev_work{ workspace_device };
            util::buffer<std::uint8_t> host_work;
            if (workspace_host)
                host_work = util::buffer<std::uint8_t>{ workspace_host };

            status = cusolverDnXgetrf(handle, nullptr, M, N,
                cuda_data_type<Real>::value,
                dev_a.get(), M,
                dev_ipiv.get(),
                cuda_data_type<Real>::value,
                dev_work.get(), workspace_device,
                host_work.get(), workspace_host,
                dev_info.get());
            if (status != CUSOLVER_STATUS_SUCCESS)
                throw std::runtime_error("cusolverDnXgetrf error");

            smp.do_sample();

            util::copy(dev_a, a.begin(), a.size());
            util::copy(dev_ipiv, ipiv.begin(), ipiv.size());
            util::copy(dev_info, &info, 1);
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

    namespace work_type
    {
        enum type
        {
            dtrtri,
            strtri,
            dgetrf,
            sgetrf,
        };
    };

    struct cmdargs
    {
        std::size_t m = 0;
        std::size_t n = 0;
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
                if (m == 0)
                    throw std::invalid_argument("m must be a positive integer");
                util::to_scalar(argv[3], n);
                if (n == 0)
                    throw std::invalid_argument("n must be a positive integer");
                assert(m > 0);
                assert(n > 0);
            }
            else if (wtype == work_type::dtrtri || wtype == work_type::strtri)
            {
                if (argc < 3)
                    throw_too_few(prog, op_type);
                util::to_scalar(argv[2], n);
                if (n == 0)
                    throw std::invalid_argument("n must be a positive integer");
                assert(n > 0);
            }
        }

    private:
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
            throw std::invalid_argument(std::string("Unknown work type ").append(op_type));
        }

        static void usage(const char* prog)
        {
            std::cerr << "Usage:\n"
                << "\t" << prog << " {dtrtri,strtri} <n>\n"
                << "\t" << prog << " {dgetrf,sgetrf} <m> <n>\n";
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
        case work_type::dgetrf:
            std::cerr << "dgetrf M=" << args.m << ", N=" << args.n << "\n";
            return dgetrf(handle, args.m, args.n, gen);
        case work_type::sgetrf:
            std::cerr << "sgetrf M=" << args.m << ", N=" << args.n << "\n";
            return sgetrf(handle, args.m, args.n, gen);
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
    }
}
