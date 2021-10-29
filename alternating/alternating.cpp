#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>

#include <cassert>
#include <chrono>
#include <random>
#include <thread>

#include <omp.h>

namespace
{
    tp::printer g_tpr;

    class timer
    {
    public:
        using time_point = std::chrono::time_point<std::chrono::steady_clock>;
        using duration = std::chrono::nanoseconds;

        timer(duration& dur) :
            _dur(dur),
            _start(time_point::clock::now())
        {}

        ~timer()
        {
            auto end = time_point::clock::now();
            _dur = std::chrono::duration_cast<duration>(end - _start);
        }

    private:
        duration& _dur;
        time_point _start;
    };

    struct cmdargs
    {
        std::chrono::nanoseconds period = {};
        std::int32_t count = 0;

        cmdargs(int argc, const char* const* argv)
        {
            const char* prog = argv[0];
            if (argc < 3)
            {
                print_usage(prog);
                throw std::invalid_argument("Not enough arguments");
            }
            period = std::chrono::milliseconds{
                util::to_scalar<decltype(period)::rep>(argv[1])
            };
            if (period.count() <= 0)
                throw std::invalid_argument("Period must be a positive integer");
            util::to_scalar(argv[2], count);
            if (count <= 0)
                throw std::invalid_argument("Count must be a positive integer");
            assert(period.count() > 0);
            assert(count > 0);
        }

    private:
        void print_usage(const char* prog)
        {
            std::cerr << "Usage: " << prog << " <interval_ms> <count>\n";
        }
    };

    void generate(
        std::vector<std::mt19937_64>& engines,
        std::size_t count,
        double lower,
        double upper)
    {
    #pragma omp parallel
        {
            auto& engine = engines[omp_get_thread_num()];
            std::uniform_real_distribution dist{ lower, upper };
        #pragma omp for
            for (std::size_t i = 0; i < count; i++)
                std::ignore = dist(engine);
        }
    }

    std::vector<std::mt19937_64> get_engines()
    {
        std::random_device rd;
        std::vector<std::mt19937_64> engines;

        int max_threads = omp_get_max_threads();
        engines.reserve(max_threads);
        for (int i = 0; i < max_threads; i++)
        {
            std::seed_seq sseq{ rd(), rd(), rd() };
            engines.emplace_back(sseq);
        }
        return engines;
    }

    std::size_t find_rng_count(
        std::vector<std::mt19937_64>& engines,
        const std::chrono::nanoseconds& period,
        double tolerance)
    {
        using nanos = std::chrono::nanoseconds;
        auto is_accurate = [](const nanos& dur, const nanos& period, double tolerance)
        {
            return std::abs((dur - period).count()) <=
                tolerance * std::chrono::duration_cast<decltype(dur - period)>(period).count();
        };

        constexpr const std::size_t start = 1000000UL;
        constexpr const std::size_t max_iters = 100UL;
        for (std::size_t i = 0, count = start; i < max_iters; i++)
        {
            nanos dur{};
            {
                timer t(dur);
                generate(engines, count, 0.0, 1.0);
            }
            if (is_accurate(dur, period, tolerance))
                return count;
            auto ratio = std::chrono::duration_cast<std::chrono::duration<double>>(period) / dur;
            count *= ratio;
        };
        throw std::runtime_error("Reached max iterations without satisfying accuracy");
    }

    __attribute__((noinline)) void do_work(
        std::vector<std::mt19937_64>& engines,
        const std::chrono::nanoseconds& period,
        std::int32_t count,
        std::size_t rng_count)
    {
        for (auto i = 0; i < count; i++)
        {
            std::this_thread::sleep_for(period);
            tp::sampler s(g_tpr);
            generate(engines, rng_count, 0.0, 1.0);
        }
    }
}

int main(int argc, char** argv)
{
    try
    {
        const cmdargs args(argc, argv);
        auto engines = get_engines();
        auto rng_count = find_rng_count(engines, args.period, 0.05);
        g_tpr.sample();
        do_work(engines, args.period, args.count, rng_count);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}