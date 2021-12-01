#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#include <cassert>
#include <random>
#include <string_view>
#include <vector>

namespace
{
    tp::printer g_tpr;

    namespace detail
    {
        template<typename T>
        struct int_distribution_traits
        {
            using value_type = T;
            using distribution = std::uniform_int_distribution<value_type>;
            static constexpr value_type lower = T{};
            static constexpr value_type upper = 100;
        };

        template<typename T>
        struct real_distribution_traits
        {
            using value_type = T;
            using distribution = std::uniform_real_distribution<value_type>;
            static constexpr value_type lower = T{};
            static constexpr value_type upper = 1.0;
        };
    };

    enum class placement
    {
        host,
        device,
    };

    enum class operation
    {
        sum,
        mult,
        min,
        max,
        count_if,
    };

    enum class element_type
    {
        int32,
        int64,
        real32,
        real64,
    };

    template<typename T>
    struct sum
    {
        using value_type = T;
        using functor = thrust::plus<value_type>;
        static constexpr value_type init_value = 0;
    };

    template<typename T>
    struct mult
    {
        using value_type = T;
        using functor = thrust::multiplies<value_type>;
        static constexpr value_type init_value = 1;
    };

    template<typename>
    struct distribution_traits {};

    template<>
    struct distribution_traits<std::int32_t> :
        detail::int_distribution_traits<std::int32_t>
    {};

    template<>
    struct distribution_traits<std::int64_t> :
        detail::int_distribution_traits<std::int64_t>
    {};

    template<>
    struct distribution_traits<float> :
        detail::real_distribution_traits<float>
    {};

    template<>
    struct distribution_traits<double> :
        detail::real_distribution_traits<double>
    {};

    template<typename T>
    struct element_traits
    {
        using value_type = T;
        using dist_traits = distribution_traits<value_type>;
        static constexpr value_type count_threshold =
            (dist_traits::lower + dist_traits::upper) / 2;
    };

    struct greater_than_threshold
    {
        template<typename T>
        __host__ __device__
            bool operator()(const T& x)
        {
            return x >= element_traits<T>::count_threshold;
        }
    };

    struct arguments
    {
        placement where = placement::host;
        operation op = operation::sum;
        element_type etype = element_type::int32;
        std::size_t n = 0;
        std::size_t iters = 0;

        arguments(int argc, const char* const* argv)
        {
            if (argc < 5)
            {
                print_usage(argv[0]);
                throw std::invalid_argument("Not enough arguments");
            }
            where = get_placement(argv[1]);
            op = get_operation(argv[2]);
            etype = get_element_type(argv[3]);
            util::to_scalar(argv[4], n);
            assert(n > 0);
            if (!n)
                throw std::invalid_argument("n must be greater than 0");
            util::to_scalar(argv[5], iters);
            assert(iters > 0);
            if (!iters)
                throw std::invalid_argument("iters must be greater than 0");
        }

    private:
        std::string lowercase(std::string_view str)
        {
            std::string lower(str);
            std::transform(lower.begin(), lower.end(), lower.begin(),
                [](unsigned char c)
                {
                    return std::tolower(c);
                });
            return lower;
        }

        placement get_placement(std::string_view arg)
        {
            std::string lower = lowercase(arg);
            if (lower == "host")
                return placement::host;
            if (lower == "device")
                return placement::device;
            throw std::invalid_argument("invalid placement");
        }

        operation get_operation(std::string_view arg)
        {
            std::string lower = lowercase(arg);
            if (lower == "sum")
                return operation::sum;
            if (lower == "mult")
                return operation::mult;
            if (lower == "min")
                return operation::min;
            if (lower == "max")
                return operation::max;
            if (lower == "count_if")
                return operation::count_if;
            throw std::invalid_argument("invalid operation");
        }

        element_type get_element_type(std::string_view arg)
        {
            std::string lower = lowercase(arg);
            if (lower == "i32")
                return element_type::int32;
            if (lower == "i64")
                return element_type::int64;
            if (lower == "r32")
                return element_type::real32;
            if (lower == "r64")
                return element_type::real64;
            throw std::invalid_argument("invalid element type");
        }

        void print_usage(const char* prog)
        {
            std::cerr << "Usage: " << prog
                << " <host|device> <operation> <type> <n> <iters>\n";
            std::cerr << "\toperation: sum | max | min | count | count_if\n";
            std::cerr << "\ttype: i32 | i64 | r32 | r64\n";
        }
    };

    template<typename T>
    __attribute__((noinline)) thrust::host_vector<T> random_vector(
        std::size_t n,
        std::mt19937_64& engine)
    {
        using traits = element_traits<T>;
        tp::sampler smp(g_tpr);
        (void)smp;
        typename traits::dist_traits::distribution dist{
            traits::dist_traits::lower, traits::dist_traits::upper
        };
        thrust::host_vector<typename traits::value_type> vec(n);
        std::generate(vec.begin(), vec.end(),
            [&]()
            {
                return dist(engine);
            });
        return vec;
    }

    template<typename BinaryOp>
    __attribute__((noinline)) typename BinaryOp::value_type reduce_host_work(
        const thrust::host_vector<typename BinaryOp::value_type>& vec,
        std::size_t iters)
    {
        using value_type = typename BinaryOp::value_type;
        tp::sampler smp(g_tpr);
        (void)smp;
        thrust::host_vector<value_type> results(iters);
        for (std::size_t i = 0; i < iters; i++)
        {
            results[i] = thrust::reduce(
                thrust::host,
                std::begin(vec),
                std::end(vec),
                BinaryOp::init_value,
                typename BinaryOp::functor{});
        }
        return results.front();
    }

    template<typename T>
    __attribute__((noinline)) T count_if_host_work(
        const thrust::host_vector<T>& vec,
        std::size_t iters)
    {
        tp::sampler smp(g_tpr);
        (void)smp;
        thrust::host_vector<T> results(iters);
        for (std::size_t i = 0; i < iters; i++)
        {
            results[i] = thrust::count_if(
                thrust::host,
                std::begin(vec),
                std::end(vec),
                greater_than_threshold{});
        }
        return results.front();
    }

    template<typename T>
    __attribute__((noinline)) T max_element_host_work(
        const thrust::host_vector<T>& vec,
        std::size_t iters)
    {
        tp::sampler smp(g_tpr);
        (void)smp;
        thrust::host_vector<T> results(iters);
        for (std::size_t i = 0; i < iters; i++)
            results[i] = *thrust::max_element(thrust::host, std::begin(vec), std::end(vec));
        return results.front();
    }

    template<typename T>
    __attribute__((noinline)) T min_element_host_work(
        const thrust::host_vector<T>& vec,
        std::size_t iters)
    {
        tp::sampler smp(g_tpr);
        (void)smp;
        thrust::host_vector<T> results(iters);
        for (std::size_t i = 0; i < iters; i++)
            results[i] = *thrust::min_element(thrust::host, std::begin(vec), std::end(vec));
        return results.front();
    }

    template<typename BinaryOp>
    __attribute__((noinline)) typename BinaryOp::value_type reduce(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<typename BinaryOp::value_type>(n, engine);
        return reduce_host_work<BinaryOp>(vec, iters);
    }

    template<typename T>
    __attribute__((noinline)) T count_if(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<T>(n, engine);
        return count_if_host_work(vec, iters);
    }

    template<typename T>
    __attribute__((noinline)) T max_element(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<T>(n, engine);
        return max_element_host_work(vec, iters);
    }

    template<typename T>
    __attribute__((noinline)) T min_element(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<T>(n, engine);
        return min_element_host_work(vec, iters);
    }

    void dispatch_work(const arguments& args, std::mt19937_64& engine)
    {
        using real32_t = float;
        using real64_t = double;
    #define DISPATCH_REDUCE(type, op_type) \
        do { \
            if (args.etype == element_type::type && args.op == operation::op_type) \
            { \
                std::cerr << #type " " #op_type "\n"; \
                std::cerr << reduce<op_type<type ## _t>>(args.n, args.iters, engine) << "\n"; \
                return; \
            } \
        } while (false)

    #define DISPATCH_REDUCE_ALL(type) \
        DISPATCH_REDUCE(type, sum); \
        DISPATCH_REDUCE(type, mult)

    #define DISPATCH_COUNT_IF(type) \
        do { \
            if (args.etype == element_type::type && args.op == operation::count_if) \
            { \
                std::cerr << #type " count_if\n"; \
                std::cerr << count_if<type ## _t>(args.n, args.iters, engine) << "\n"; \
                return; \
            } \
        } while (false)

    #define DISPATCH_EXTREMA(type, op_type) \
        do { \
            if (args.etype == element_type::type && args.op == operation::op_type) \
            { \
                std::cerr << #type " " #op_type "\n"; \
                std::cerr << op_type ## _element<type ## _t>(args.n, args.iters, engine) << "\n"; \
                return; \
            } \
        } while (false)

    #define DISPATCH_EXTREMA_ALL(type) \
        DISPATCH_EXTREMA(type, max); \
        DISPATCH_EXTREMA(type, min)

        DISPATCH_REDUCE_ALL(int32);
        DISPATCH_REDUCE_ALL(int64);
        DISPATCH_REDUCE_ALL(real32);
        DISPATCH_REDUCE_ALL(real64);
        DISPATCH_COUNT_IF(int32);
        DISPATCH_COUNT_IF(int64);
        DISPATCH_COUNT_IF(real32);
        DISPATCH_COUNT_IF(real64);
        DISPATCH_EXTREMA_ALL(int32);
        DISPATCH_EXTREMA_ALL(int64);
        DISPATCH_EXTREMA_ALL(real32);
        DISPATCH_EXTREMA_ALL(real64);
        throw std::runtime_error("unable to dispatch work");
    }
}

int main(int argc, char** argv)
{
    try
    {
        const arguments args{ argc, argv };
        std::random_device rnd_dev;
        std::mt19937_64 engine{ rnd_dev() };
        dispatch_work(args, engine);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
