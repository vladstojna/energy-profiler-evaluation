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

    struct host_placement
    {
        template<typename T>
        using container = thrust::host_vector<T>;
        static const decltype(thrust::host) policy;
    };

    struct device_placement
    {
        template<typename T>
        using container = thrust::device_vector<T>;
        static const decltype(thrust::device) policy;
    };

    decltype(host_placement::policy) host_placement::policy = thrust::host;
    decltype(device_placement::policy) device_placement::policy = thrust::device;

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
        __host__ __device__ bool operator()(const std::int32_t& x) { return call_impl(x); }
        __host__ __device__ bool operator()(const std::int64_t& x) { return call_impl(x); }
        __host__ __device__ bool operator()(const float& x) { return call_impl(x); }
        __host__ __device__ bool operator()(const double& x) { return call_impl(x); }

    private:
        template<typename T>
        __host__ __device__
            bool call_impl(const T& x)
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
            if (argc < 6)
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
        std::string lowercase(const char* str)
        {
            std::string lower(str);
            auto tolower = [](unsigned char c) { return std::tolower(c); };
            for (char& c : lower)
                c = tolower(c);
            return lower;
        }

        placement get_placement(const char* arg)
        {
            std::string lower = lowercase(arg);
            if (lower == "host")
                return placement::host;
            if (lower == "device")
                return placement::device;
            throw std::invalid_argument("invalid placement");
        }

        operation get_operation(const char* arg)
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

        element_type get_element_type(const char* arg)
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
            std::cerr << "\toperation: sum | mult | min | max | count_if\n";
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
        for (typename traits::value_type& val : vec)
            val = dist(engine);
        return vec;
    }

    template<typename Placement, typename BinaryOp>
    __attribute__((noinline)) typename BinaryOp::value_type reduce_work(
        const typename Placement::container<typename BinaryOp::value_type>& vec,
        std::size_t iters)
    {
        using value_type = typename BinaryOp::value_type;
        tp::sampler smp(g_tpr);
        (void)smp;
        thrust::host_vector<value_type> results(iters);
        for (std::size_t i = 0; i < iters; i++)
            results[i] = thrust::reduce(
                Placement::policy,
                std::begin(vec),
                std::end(vec),
                BinaryOp::init_value,
                typename BinaryOp::functor{});
        return results.front();
    }

    template<typename Placement, typename T>
    __attribute__((noinline)) T count_if_work(
        const typename Placement::container<T>& vec,
        std::size_t iters)
    {
        tp::sampler smp(g_tpr);
        (void)smp;
        thrust::host_vector<T> results(iters);
        for (std::size_t i = 0; i < iters; i++)
            results[i] = thrust::count_if(
                Placement::policy,
                std::begin(vec),
                std::end(vec),
                greater_than_threshold{});
        return results.front();
    }

    template<typename Placement, typename T>
    __attribute__((noinline)) T max_element_work(
        const typename Placement::container<T>& vec,
        std::size_t iters)
    {
        tp::sampler smp(g_tpr);
        (void)smp;
        thrust::host_vector<T> results(iters);
        for (std::size_t i = 0; i < iters; i++)
            results[i] = *thrust::max_element(
                Placement::policy, std::begin(vec), std::end(vec));
        return results.front();
    }

    template<typename Placement, typename T>
    __attribute__((noinline)) T min_element_work(
        const typename Placement::container<T>& vec,
        std::size_t iters)
    {
        tp::sampler smp(g_tpr);
        (void)smp;
        thrust::host_vector<T> results(iters);
        for (std::size_t i = 0; i < iters; i++)
            results[i] = *thrust::min_element(
                Placement::policy, std::begin(vec), std::end(vec));
        return results.front();
    }

    template<typename BinaryOp>
    __attribute__((noinline)) typename BinaryOp::value_type reduce_host(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<typename BinaryOp::value_type>(n, engine);
        return reduce_work<host_placement, BinaryOp>(vec, iters);
    }

    template<typename BinaryOp>
    __attribute__((noinline)) typename BinaryOp::value_type reduce_device(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<typename BinaryOp::value_type>(n, engine);
        thrust::device_vector<typename BinaryOp::value_type> d_vec = vec;
        return reduce_work<device_placement, BinaryOp>(d_vec, iters);
    }

    template<typename T>
    __attribute__((noinline)) T count_if_host(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<T>(n, engine);
        return count_if_work<host_placement>(vec, iters);
    }

    template<typename T>
    __attribute__((noinline)) T count_if_device(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<T>(n, engine);
        thrust::device_vector<T> d_vec = vec;
        return count_if_work<device_placement>(d_vec, iters);
    }

    template<typename T>
    __attribute__((noinline)) T max_element_host(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<T>(n, engine);
        return max_element_work<host_placement>(vec, iters);
    }

    template<typename T>
    __attribute__((noinline)) T max_element_device(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<T>(n, engine);
        thrust::device_vector<T> d_vec = vec;
        return max_element_work<device_placement>(d_vec, iters);
    }

    template<typename T>
    __attribute__((noinline)) T min_element_host(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<T>(n, engine);
        return min_element_work<host_placement>(vec, iters);
    }

    template<typename T>
    __attribute__((noinline)) T min_element_device(
        std::size_t n,
        std::size_t iters,
        std::mt19937_64& engine)
    {
        auto vec = random_vector<T>(n, engine);
        thrust::device_vector<T> d_vec = vec;
        return min_element_work<device_placement>(d_vec, iters);
    }

    void dispatch_work(const arguments& args, std::mt19937_64& engine)
    {
    #define DISPATCH_ALL_TYPES(macro) \
        macro(int32); \
        macro(int64); \
        macro(real32); \
        macro(real64)

    #define DISPATCH_REDUCE(place, type, op_type) \
        do { \
            if (args.where == placement::place && \
                args.etype == element_type::type && \
                args.op == operation::op_type) \
            { \
                std::cerr << #place " " #type " " #op_type "\n"; \
                std::cerr << reduce_ ##place<op_type<type ## _t>>(args.n, args.iters, engine) << "\n"; \
                return; \
            } \
        } while (false)

    #define DISPATCH_REDUCE_ALL(type) \
        DISPATCH_REDUCE(host, type, sum); \
        DISPATCH_REDUCE(host, type, mult); \
        DISPATCH_REDUCE(device, type, sum); \
        DISPATCH_REDUCE(device, type, mult)

    #define DISPATCH_COUNT_IF(place, type) \
        do { \
            if (args.where == placement::place && \
                args.etype == element_type::type && \
                args.op == operation::count_if) \
            { \
                std::cerr << #place " " #type " count_if\n"; \
                std::cerr << count_if_ ##place<type ## _t>(args.n, args.iters, engine) << "\n"; \
                return; \
            } \
        } while (false)

    #define DISPATCH_COUNT_IF_ALL(type) \
        DISPATCH_COUNT_IF(host, type); \
        DISPATCH_COUNT_IF(device, type)

    #define DISPATCH_EXTREMA(place, type, op_type) \
        do { \
            if (args.where == placement::place && \
                args.etype == element_type::type && \
                args.op == operation::op_type) \
            { \
                std::cerr << #place " " #type " " #op_type "\n"; \
                std::cerr << op_type ## _element_ ##place<type ## _t>(args.n, args.iters, engine) \
                    << "\n"; \
                return; \
            } \
        } while (false)

    #define DISPATCH_EXTREMA_ALL(type) \
        DISPATCH_EXTREMA(host, type, max); \
        DISPATCH_EXTREMA(host, type, min); \
        DISPATCH_EXTREMA(device, type, max); \
        DISPATCH_EXTREMA(device, type, min)

        using real32_t = float;
        using real64_t = double;
        DISPATCH_ALL_TYPES(DISPATCH_REDUCE_ALL);
        DISPATCH_ALL_TYPES(DISPATCH_COUNT_IF_ALL);
        DISPATCH_ALL_TYPES(DISPATCH_EXTREMA_ALL);
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
