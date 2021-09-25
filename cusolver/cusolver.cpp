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

    namespace work_type
    {
        enum type
        {
            dtrtri,
            strtri
        };
    };

    struct cmdargs
    {
        std::size_t n = 0;
        work_type::type wtype = static_cast<work_type::type>(0);

        cmdargs(int argc, const char* const* argv)
        {
            if (argc < 3)
            {
                usage(argv[0]);
                throw std::invalid_argument("Not enough arguments");
            }
            wtype = get_work_type(argv[1]);
            util::to_scalar(argv[2], n);
            if (n == 0)
                throw std::invalid_argument("n must be a positive integer");
            assert(n > 0);
            assert(wtype);
        }

    private:
        static work_type::type get_work_type(const char* str)
        {
            std::string op_type(str);
            std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                [](unsigned char c) { return std::tolower(c); });

            if (op_type == "dtrtri")
                return work_type::dtrtri;
            if (op_type == "strtri")
                return work_type::strtri;
            throw std::invalid_argument(std::string("Unknown work type ").append(op_type));
        }

        static void usage(const char* prog)
        {
            std::cerr << "Usage: " << prog << " {dtrtri,strtri} <n>\n";
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
