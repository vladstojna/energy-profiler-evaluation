#include <timeprinter/printer.hpp>

#include <condition_variable>
#include <mutex>
#include <vector>
#include <thread>

namespace tp
{
    using time_point = std::chrono::time_point<std::chrono::steady_clock>;

    struct printer::impl
    {
        std::ostream& os;
        std::string ctx;

        impl(std::string_view ctx, std::ostream& os) :
            os(os),
            ctx(ctx)
        {}

        virtual ~impl() = default;
        virtual void sample() = 0;
    };
}

namespace
{
    std::chrono::duration<double>
        get_duration(const tp::time_point& start, const tp::time_point& end)
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    }

    struct simple_printer : tp::printer::impl
    {
        std::vector<tp::time_point> samples;

        simple_printer(std::string_view ctx, std::ostream& os) :
            impl(ctx, os)
        {
            samples.reserve(2);
            samples.push_back(tp::time_point::clock::now());
        };

        void sample() override
        {
            samples.push_back(tp::time_point::clock::now());
        }

        ~simple_printer()
        {
            auto end = tp::time_point::clock::now();
            os << "#context," << ctx << "\n";
            os << "#duration,s," << get_duration(samples.front(), end).count() << "\n";
            os << "count,time\n";
            for (std::size_t ix = 0; ix < samples.size(); ix++)
                os << ix << "," << samples[ix].time_since_epoch().count() << "\n";
            os << samples.size() << "," << end.time_since_epoch().count() << "\n";
        };
    };

    struct periodic_printer : tp::printer::impl
    {
        bool finished;
        std::condition_variable cv;
        std::mutex mtx;
        std::thread thread;

        std::vector<tp::time_point> samples;

        periodic_printer(std::string_view ctx, const tp::period_data& data, std::ostream& os) :
            impl(ctx, os),
            finished(false)
        {
            samples.reserve(data.initial_size);
            thread = std::thread(&periodic_printer::thread_func, this, data.interval);
            {
                std::scoped_lock lk(mtx);
                samples.push_back(tp::time_point::clock::now());
            }
        };

        void sample() override
        {
            std::scoped_lock lk(mtx);
            samples.push_back(tp::time_point::clock::now());
        }

        ~periodic_printer()
        {
            {
                std::scoped_lock lk(mtx);
                samples.push_back(tp::time_point::clock::now());
                finished = true;
            }
            cv.notify_one();
            thread.join();
            os << "#context," << ctx << "\n";
            os << "#duration,s," << get_duration(samples.front(), samples.back()).count() << "\n";
            os << "count,time\n";
            for (std::size_t ix = 0; ix < samples.size(); ix++)
                os << ix << "," << samples[ix].time_since_epoch().count() << "\n";
        };

    private:
        void thread_func(const tp::time_point::duration& interval)
        {
            std::unique_lock lk(mtx);
            if (finished)
                return;
            while (true)
            {
                cv.wait_for(lk, interval);
                if (finished)
                    return;
                samples.push_back(tp::time_point::clock::now());
            }
        }
    };

    std::unique_ptr<tp::printer::impl> make_printer(
        std::string_view context,
        const std::optional<tp::period_data>& periodic,
        std::ostream& os)
    {
        if (periodic)
            return std::make_unique<periodic_printer>(context, *periodic, os);
        else
            return std::make_unique<simple_printer>(context, os);
    }
}

namespace tp
{
    printer::printer(
        std::string_view context,
        const std::optional<period_data>& periodic,
        std::ostream& os)
        :
        _impl(make_printer(context, periodic, os))
    {}

    printer::printer(std::string_view context, std::ostream& os) :
        printer(context, std::nullopt, os)
    {}

    printer::printer(const std::optional<period_data>& periodic, std::ostream& os) :
        printer("", periodic, os)
    {}

    printer::printer(std::ostream& os) :
        printer(std::nullopt, os)
    {}

    printer::~printer() = default;

    void printer::sample()
    {
        _impl->sample();
    }
}
