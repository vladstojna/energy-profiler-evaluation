#include <util/unique_handle.hpp>
#include <util/buffer.hpp>
#include <util/cuda_utils.hpp>
#include <util/to_scalar.hpp>
#include <timeprinter/printer.hpp>

#include <cuda.h>
#include <curand.h>
#include <omp.h>

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

    void curand_generator_destroy(curandGenerator_t gen)
    {
        auto status = curandDestroyGenerator(gen);
        if (status != CURAND_STATUS_SUCCESS)
            std::cerr << "Error destroying cuRAND generator\n";
    }

    void curand_set_generator_seed(curandGenerator_t gen, std::random_device& rd)
    {
        auto status = curandSetPseudoRandomGeneratorSeed(gen,
            static_cast<std::uint64_t>(rd()) << 32 | rd());
        if (status != CURAND_STATUS_SUCCESS)
            throw std::runtime_error("Error setting pseudo random generator seed");
    }

    std::vector<std::mt19937> get_engines(std::random_device& rd)
    {
        std::vector<std::mt19937> engines;
        int max_threads = omp_get_max_threads();
        engines.reserve(max_threads);
        for (int i = 0; i < max_threads; i++)
        {
            std::seed_seq sseq{ rd(), rd(), rd() };
            engines.emplace_back(sseq);
        }
        return engines;
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
        typename util::buffer<RealType>::size_type
        generate_host(std::size_t count, std::size_t iters, std::vector<std::mt19937>& engines)
    {
        assert(count > 0);
        tp::sampler smp(g_tpr);
        util::buffer<RealType> host_buff{ count };
        smp.do_sample();
    #pragma omp parallel
        {
            auto& engine = engines[omp_get_thread_num()];
            std::uniform_real_distribution<RealType> dist{ 0.0, 1.0 };
            for (std::size_t x = 0; x < iters; x++)
            {
            #pragma omp for
                for (std::size_t i = 0; i < host_buff.size(); i++)
                {
                    host_buff[i] = dist(engine);
                }
            }
        }
        return host_buff.size();
    }

    template<typename RealType>
    __attribute__((noinline))
        typename util::device_buffer<RealType>::size_type
        generate_device(std::size_t count, std::size_t iters, curandGenerator_t gen)
    {
        assert(count > 0);
        tp::sampler smp(g_tpr);
        util::device_buffer<RealType> dev_buff{ count };
        smp.do_sample();
        for (std::size_t x = 0; x < iters; x++)
        {
            auto status = detail::generate_func<RealType>{}(gen, dev_buff.get(), count);
            if (status != CURAND_STATUS_SUCCESS)
                throw std::runtime_error("Error generating uniform distribution");
            cudaDeviceSynchronize();
        }
        return dev_buff.size();
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
        std::size_t iters;
        work_type::type wtype;

        cmdargs(int argc, const char* const* argv) :
            count(0),
            iters(1),
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
            if (argc > 4)
            {
                util::to_scalar(argv[4], iters);
                if (!iters)
                    throw std::invalid_argument("[iters] must be positive");
            }
            assert(wtype);
            assert(count);
            assert(iters >= 1);
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
            std::cerr << "Usage: " << prog << " <count> {host,device} {d,s} [iters]\n";
        }
    };
}

int main(int argc, char** argv)
{
    try
    {
        auto execute_work = [](const cmdargs& args, std::random_device& rd)
        {
            switch (args.wtype)
            {
            case work_type::device | work_type::double_prec:
            {
                generator_handle gen{ curand_generator_create_device() };
                curand_set_generator_seed(gen, rd);
                return generate_device<double>(args.count, args.iters, gen);
            }
            case work_type::device | work_type::single_prec:
            {
                generator_handle gen{ curand_generator_create_device() };
                curand_set_generator_seed(gen, rd);
                return generate_device<float>(args.count, args.iters, gen);
            }
            case work_type::host | work_type::double_prec:
            {
                auto engines = get_engines(rd);
                return generate_host<double>(args.count, args.iters, engines);
            }
            case work_type::host | work_type::single_prec:
            {
                auto engines = get_engines(rd);
                return generate_host<float>(args.count, args.iters, engines);
            }
            }
            throw std::runtime_error("execute_work: invalid work type");
        };

        const cmdargs args(argc, argv);
        std::random_device rd;
        auto count = execute_work(args, rd);
        std::cerr << "Generated " << count << " pseudorandom numbers\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}