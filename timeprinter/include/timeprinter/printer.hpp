#pragma once

#include <chrono>
#include <iostream>
#include <memory>

#if __cplusplus < 201402L
#error "C++14 or better is required"
#endif

#if __cplusplus >= 201703L
#include <optional>
#include <string_view>
#endif

namespace tp {
struct period_data {
  std::chrono::nanoseconds interval;
  std::size_t initial_size = 2;

#if __cplusplus < 201703L
  explicit operator bool() const { return interval.count() && initial_size; }
#endif
};

#if __cplusplus >= 201703L
using context_type = std::string_view;
using opt_period_data = std::optional<period_data>;
static constexpr auto no_period = std::nullopt;
#else
using context_type = const char *;
using opt_period_data = period_data;
static constexpr auto no_period =
    opt_period_data{std::chrono::nanoseconds{}, {}};
#endif

class printer {
public:
  struct impl;

  printer(std::ostream & = std::cout);
  printer(const opt_period_data &, std::ostream & = std::cout);
  printer(context_type, std::ostream & = std::cout);
  printer(context_type, const opt_period_data &, std::ostream & = std::cout);

  ~printer();

  void sample();

  printer(const printer &) = delete;
  printer(printer &&) = delete;
  printer &operator=(const printer &) = delete;
  printer &operator=(printer &&) = delete;

private:
  std::unique_ptr<impl> _impl;
};

class sampler {
public:
  sampler(printer &p) : _p(p) { _p.sample(); }

  ~sampler() { _p.sample(); }

  void do_sample() { _p.sample(); }

private:
  printer &_p;
};
} // namespace tp
