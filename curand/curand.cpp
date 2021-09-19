#include <util/unique_handle.hpp>
#include <util/buffer.hpp>
#include <util/cuda_utils.hpp>
#include <util/to_scalar.hpp>
#include <timeprinter/printer.hpp>

#include <cuda.h>
#include <curand.h>

#include <cassert>
#include <iostream>
#include <random>

namespace
{
    tp::printer g_tpr;

    template<auto func>
    struct func_obj : std::integral_constant<decltype(func), func>
    {};

    curandGenerator_t curand_generator_create_device()
    {
        curandGenerator_t gen;
        auto status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
        if (status != CURAND_STATUS_SUCCESS)
            throw std::runtime_error("Error creating device generator");
        return gen;
    }

    curandGenerator_t curand_generator_create_host()
    {
        curandGenerator_t gen;
        auto status = curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_MT19937);
        if (status != CURAND_STATUS_SUCCESS)
            throw std::runtime_error("Error creating device generator");
        return gen;
    }

    void curand_generator_destroy(curandGenerator_t gen)
    {
        auto status = curandDestroyGenerator(gen);
        if (status != CURAND_STATUS_SUCCESS)
            std::cerr << "Error destroying cuRAND generator\n";
    }

    void curand_set_generator_seed(curandGenerator_t gen, unsigned long long seed)
    {
        auto status = curandSetPseudoRandomGeneratorSeed(gen, seed);
        if (status != CURAND_STATUS_SUCCESS)
            throw std::runtime_error("Error setting pseudo random generator seed");
    }

    using generator_handle = util::unique_handle<
        curandGenerator_t,
        func_obj<curand_generator_destroy>
    >;

    namespace detail
    {
        template<typename T>
        struct generate_func {};

        template<>
        struct generate_func<double>
        {
            auto operator()(curandGenerator_t gen, double* buff, std::size_t count)
            {
                return curandGenerateUniformDouble(gen, buff, count);
            }
        };

        template<>
        struct generate_func<float>
        {
            auto operator()(curandGenerator_t gen, float* buff, std::size_t count)
            {
                return curandGenerateUniform(gen, buff, count);
            }
        };
    }

    template<typename RealType>
    __attribute__((noinline))
        typename util::host_buffer<RealType>::size_type
        generate_host(std::size_t count, curandGenerator_t gen)
    {
        using namespace util;
        assert(count > 0);
        tp::sampler smp(g_tpr);
        host_buffer<RealType> host_buff{ count };
        smp.do_sample();
        auto status = detail::generate_func<RealType>{}(gen, host_buff.get(), count);
        if (status != CURAND_STATUS_SUCCESS)
            throw std::runtime_error("Error generating uniform distribution");
        return host_buff.size();
    }

    template<typename RealType>
    __attribute__((noinline))
        typename util::host_buffer<RealType>::size_type
        generate_device(std::size_t count, curandGenerator_t gen)
    {
        using namespace util;
        assert(count > 0);
        tp::sampler smp(g_tpr);
        device_buffer<RealType> dev_buff{ count };
        smp.do_sample();
        auto status = detail::generate_func<RealType>{}(gen, dev_buff.get(), count);
        if (status != CURAND_STATUS_SUCCESS)
            throw std::runtime_error("Error generating uniform distribution");
        cudaDeviceSynchronize();
        smp.do_sample();
        return host_buffer{ dev_buff }.size();
    }

    namespace work_type
    {
        using type = std::uint32_t;
        constexpr const type none = 0x0;
        constexpr const type host = 0x1 << 0;
        constexpr const type device = 0x1 << 1;
        constexpr const type double_prec = 0x1 << 2;
        constexpr const type single_prec = 0x1 << 3;
    }

    struct cmdargs
    {
        std::size_t count;
        work_type::type wtype;

        cmdargs(int argc, const char* const* argv) :
            count(0),
            wtype(work_type::none)
        {
            if (argc < 4)
            {
                usage(argv[0]);
                throw std::invalid_argument("Not enough arguments");
            }
            util::to_scalar(argv[1], count);
            if (count == 0)
                throw std::invalid_argument("<count> must be positive");

            std::string run_on = lower_case_arg(argv[2]);
            std::string precision = lower_case_arg(argv[3]);
            if (run_on == "host")
                wtype |= work_type::host;
            else if (run_on == "device")
                wtype |= work_type::device;
            else
                throw std::invalid_argument(run_on.append(": invalid setting to run on"));
            if (precision == "d")
                wtype |= work_type::double_prec;
            else if (precision == "s")
                wtype |= work_type::single_prec;
            else
                throw std::invalid_argument(run_on.append(": invalid precision"));
            assert(wtype);
            assert(count);
        }

    private:
        std::string lower_case_arg(const char* arg)
        {
            std::string retval = arg;
            std::transform(retval.begin(), retval.end(), retval.begin(),
                [](unsigned char c) { return std::tolower(c); });
            return arg;
        }

        void usage(const char* prog)
        {
            std::cerr << "Usage: " << prog << " <count> {host,device} {d,s}\n";
        }
    };
}

int main(int argc, char** argv)
{
    try
    {
        constexpr const auto seed = 85503686ULL;

        auto create_generator = [](work_type::type wtype)
        {
            switch (wtype & (work_type::host | work_type::device))
            {
            case work_type::host:
                return generator_handle(curand_generator_create_host());
            case work_type::device:
                return generator_handle(curand_generator_create_device());
            }
            throw std::runtime_error("create_generator: invalid work type");
        };

        auto execute_work = [](work_type::type wtype, std::size_t count, curandGenerator_t gen)
        {
            switch (wtype)
            {
            case work_type::device | work_type::double_prec:
                return generate_device<double>(count, gen);
            case work_type::device | work_type::single_prec:
                return generate_device<float>(count, gen);
            case work_type::host | work_type::double_prec:
                return generate_host<double>(count, gen);
            case work_type::host | work_type::single_prec:
                return generate_host<float>(count, gen);
            }
            throw std::runtime_error("execute_work: invalid work type");
        };

        cmdargs args(argc, argv);
        auto gen = create_generator(args.wtype);
        curand_set_generator_seed(gen, seed);
        auto count = execute_work(args.wtype, args.count, gen);
        std::cerr << "Generated " << count << " pseudorandom numbers\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}