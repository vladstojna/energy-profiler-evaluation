#include <timeprinter/printer.hpp>

#include <condition_variable>
#include <mutex>
#include <vector>
#include <thread>

namespace tp
{
    using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

    struct printer::impl
    {
        std::ostream& os;

        impl(std::ostream& os) :
            os(os)
        {}

        virtual ~impl() = default;
    };
}

namespace
{
    struct simple_printer : tp::printer::impl
    {
        tp::time_point start;

        simple_printer(std::ostream& os) :
            impl(os),
            start(tp::time_point::clock::now())
        {};

        ~simple_printer()
        {
            auto end = tp::time_point::clock::now();
            os << "count,time\n";
            os << 0 << "," << start.time_since_epoch().count() << "\n";
            os << 1 << "," << end.time_since_epoch().count() << "\n";
        };
    };

    struct periodic_printer : tp::printer::impl
    {
        bool finished;
        std::condition_variable cv;
        std::mutex mtx;
        std::thread thread;

        std::vector<tp::time_point> samples;

        periodic_printer(const tp::period_data& data, std::ostream& os) :
            impl(os),
            finished(false)
        {
            samples.reserve(data.initial_size);
            thread = std::thread(&periodic_printer::thread_func, this, data.interval);
            {
                std::scoped_lock lk(mtx);
                samples.push_back(tp::time_point::clock::now());
            }
        };

        ~periodic_printer()
        {
            {
                std::scoped_lock lk(mtx);
                samples.push_back(tp::time_point::clock::now());
                finished = true;
            }
            cv.notify_one();
            thread.join();
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

    std::unique_ptr<tp::printer::impl>
        make_printer(const std::optional<tp::period_data>& periodic, std::ostream& os)
    {
        if (periodic)
            return std::make_unique<periodic_printer>(*periodic, os);
        else
            return std::make_unique<simple_printer>(os);
    }
}

namespace tp
{
    printer::printer(const std::optional<period_data>& periodic, std::ostream& os) :
        _impl(make_printer(periodic, os))
    {}

    printer::printer(std::ostream& os) :
        printer(std::nullopt, os)
    {}

    printer::~printer() = default;
}
