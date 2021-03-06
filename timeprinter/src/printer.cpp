#include <timeprinter/printer.hpp>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace tp {
#ifdef TP_USE_SYSTEM_CLOCK
using time_point = std::chrono::time_point<std::chrono::system_clock>;
#else
using time_point = std::chrono::time_point<std::chrono::steady_clock>;
#endif

struct printer::impl {
  std::ostream &os;
  std::string ctx;

  impl(context_type ctx, std::ostream &os) : os(os), ctx(ctx) {}

  virtual ~impl() = default;
  virtual void sample() = 0;
};
} // namespace tp

namespace {
std::chrono::duration<double> get_duration(const tp::time_point &start,
                                           const tp::time_point &end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
}

struct simple_printer : tp::printer::impl {
  std::vector<tp::time_point> samples;

  simple_printer(tp::context_type ctx, std::ostream &os) : impl(ctx, os) {
    samples.reserve(2);
    samples.push_back(tp::time_point::clock::now());
  };

  void sample() override { samples.push_back(tp::time_point::clock::now()); }

  ~simple_printer() {
    auto end = tp::time_point::clock::now();
    os << "#context," << ctx << "\n";
    os << "#duration,s," << get_duration(samples.front(), end).count() << "\n";
    os << "count,time\n";
    for (std::size_t ix = 0; ix < samples.size(); ix++)
      os << ix << "," << samples[ix].time_since_epoch().count() << "\n";
    os << samples.size() << "," << end.time_since_epoch().count() << "\n";
  };
};

struct periodic_printer : tp::printer::impl {
  bool finished;
  std::condition_variable cv;
  std::mutex mtx;
  std::thread thread;

  std::vector<tp::time_point> samples;

  periodic_printer(tp::context_type ctx, const tp::period_data &data,
                   std::ostream &os)
      : impl(ctx, os), finished(false) {
    samples.reserve(data.initial_size);
    thread = std::thread(&periodic_printer::thread_func, this, data.interval);
    {
      std::lock_guard<std::mutex> lk(mtx);
      samples.push_back(tp::time_point::clock::now());
    }
  };

  void sample() override {
    std::lock_guard<std::mutex> lk(mtx);
    samples.push_back(tp::time_point::clock::now());
  }

  ~periodic_printer() {
    {
      std::lock_guard<std::mutex> lk(mtx);
      samples.push_back(tp::time_point::clock::now());
      finished = true;
    }
    cv.notify_one();
    thread.join();
    os << "#context," << ctx << "\n";
    os << "#duration,s,"
       << get_duration(samples.front(), samples.back()).count() << "\n";
    os << "count,time\n";
    for (std::size_t ix = 0; ix < samples.size(); ix++)
      os << ix << "," << samples[ix].time_since_epoch().count() << "\n";
  };

private:
  void thread_func(const tp::time_point::duration &interval) {
    std::unique_lock<std::mutex> lk(mtx);
    if (finished)
      return;
    while (true) {
      cv.wait_for(lk, interval);
      if (finished)
        return;
      samples.push_back(tp::time_point::clock::now());
    }
  }
};

#if __cplusplus >= 201703L
tp::period_data get_period_data(const tp::opt_period_data &pd) { return *pd; }
#else
tp::period_data get_period_data(const tp::opt_period_data &pd) { return pd; }
#endif

std::unique_ptr<tp::printer::impl>
make_printer(tp::context_type context, const tp::opt_period_data &periodic,
             std::ostream &os) {
  if (periodic)
    return std::make_unique<periodic_printer>(context,
                                              get_period_data(periodic), os);
  else
    return std::make_unique<simple_printer>(context, os);
}
} // namespace

namespace tp {
printer::printer(context_type context, const opt_period_data &periodic,
                 std::ostream &os)
    : _impl(make_printer(context, periodic, os)) {}

printer::printer(context_type context, std::ostream &os)
    : printer(context, no_period, os) {}

printer::printer(const opt_period_data &periodic, std::ostream &os)
    : printer("", periodic, os) {}

printer::printer(std::ostream &os) : printer(no_period, os) {}

printer::~printer() = default;

void printer::sample() { _impl->sample(); }
} // namespace tp
