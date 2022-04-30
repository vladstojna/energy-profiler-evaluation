#pragma once

#if __cplusplus >= 201703L
#include <charconv>
#include <string_view>
#include <system_error>
#endif // __cplusplus >= 201703L

#if __GNUC__ < 11 || __cplusplus < 201703L
#include <cstdlib>
#include <locale>
#endif // __GNUC__ < 11 || __cplusplus < 201703L

#if __cplusplus < 201703L
#include <limits>
#include <type_traits>
#endif // __cplusplus < 201703L

namespace util {
#if __cplusplus >= 201703L
// std::from_chars is only available in C++17 and later
template <typename T> void to_scalar(std::string_view str, T &value) {
  auto [dummy, ec] = std::from_chars(str.begin(), str.end(), value);
  (void)dummy;
  if (auto code = std::make_error_code(ec))
    throw std::system_error(code);
}
#endif // __cplusplus >= 201703L

#if __GNUC__ < 11 || __cplusplus < 201703L
// specialize floating point conversions if GCC version < 11
// if C++ is older than 17 then implement conversion for all supported scalar
// types
namespace detail {
template <typename T> struct conversion_func {};
template <> struct conversion_func<float> {
  static constexpr auto func = std::strtof;
};
template <> struct conversion_func<double> {
  static constexpr auto func = std::strtod;
};
template <> struct conversion_func<long double> {
  static constexpr auto func = std::strtold;
};

#if __cplusplus >= 201703L
// In C++17 and later std::string_view is passed which is not guaranteed to be
// null-terminated, thus a temporary copy is necessary since C standard
// scalar conversion functions expect a null-terminated string
template <typename Real>
void to_scalar_impl(std::string_view str, Real &value) {
  std::string tmp(str);
  std::locale loc;
  std::locale::global(std::locale::classic());
  errno = 0;
  Real val = conversion_func<Real>::func(tmp.c_str(), nullptr);
  if (errno) {
    std::locale::global(loc);
    throw std::system_error(
        std::error_code(static_cast<int>(errno), std::generic_category()));
  }
  std::locale::global(loc);
  value = val;
}
#else
// Implementation available in
// https://en.cppreference.com/w/cpp/types/conjunction
template <typename...> struct conjunction : std::true_type {};
template <typename T> struct conjunction<T> : T {};
template <typename T, typename... Rest>
struct conjunction<T, Rest...>
    : std::conditional<bool(T::value), conjunction<Rest...>, T>::type {};

struct conv_long {
  using rettype = long;
  static constexpr auto func = std::strtol;
};

struct conv_ulong {
  using rettype = unsigned long;
  static constexpr auto func = std::strtoul;
};

template <> struct conversion_func<char> : conv_long {};
template <> struct conversion_func<short> : conv_long {};
template <> struct conversion_func<int> : conv_long {};
template <> struct conversion_func<unsigned char> : conv_ulong {};
template <> struct conversion_func<unsigned short> : conv_ulong {};
template <> struct conversion_func<unsigned int> : conv_ulong {};

template <> struct conversion_func<long> {
  static constexpr auto func = std::strtol;
};
template <> struct conversion_func<unsigned long> {
  static constexpr auto func = std::strtoul;
};
template <> struct conversion_func<long long> {
  static constexpr auto func = std::strtoll;
};
template <> struct conversion_func<unsigned long long> {
  static constexpr auto func = std::strtoull;
};

template <typename Scalar, typename... Args>
void to_scalar_impl(const char *str, Scalar &value, Args... args) {
  std::locale loc;
  std::locale::global(std::locale::classic());
  errno = 0;
  Scalar val = conversion_func<Scalar>::func(str, nullptr, args...);
  if (errno) {
    std::locale::global(loc);
    throw std::system_error(
        std::error_code(static_cast<int>(errno), std::generic_category()));
  }
  std::locale::global(loc);
  value = val;
}

template <typename SmallerInt, typename BiggerInt,
          typename std::enable_if<conjunction<std::is_signed<SmallerInt>,
                                              std::is_signed<BiggerInt>>::value,
                                  bool>::type = true>
bool is_in_range(BiggerInt x) {
  return x >= std::numeric_limits<SmallerInt>::min() &&
         x <= std::numeric_limits<SmallerInt>::max();
}

template <
    typename SmallerInt, typename BiggerInt,
    typename std::enable_if<conjunction<std::is_unsigned<SmallerInt>,
                                        std::is_unsigned<BiggerInt>>::value,
                            bool>::type = true>
bool is_in_range(BiggerInt x) {
  return x <= std::numeric_limits<SmallerInt>::max();
}

template <typename Integer, typename... Args>
void to_scalar_impl_overflow(const char *str, Integer &value, Args... args) {
  typename conversion_func<Integer>::rettype tmp;
  to_scalar_impl(str, tmp, args...);
  if (!is_in_range<Integer>(tmp))
    throw std::system_error(
        std::error_code(static_cast<int>(ERANGE), std::generic_category()));
  Integer smaller = tmp;
  value = smaller;
}
#endif // __cplusplus >= 201703L
} // namespace detail

#if __cplusplus >= 201703L
template <> void to_scalar(std::string_view str, float &value) {
  detail::to_scalar_impl(str, value);
}

template <> void to_scalar(std::string_view str, double &value) {
  detail::to_scalar_impl(str, value);
}

template <> void to_scalar(std::string_view str, long double &value) {
  detail::to_scalar_impl(str, value);
}
#else
template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              bool>::type = true>
void to_scalar(const char *str, T &value) {
  detail::to_scalar_impl(str, value);
}

template <typename T, typename std::enable_if<!std::is_floating_point<T>::value,
                                              bool>::type = true>
void to_scalar(const char *str, T &value) {
  detail::to_scalar_impl_overflow(str, value, 10);
}

template <> void to_scalar(const char *str, long &value) {
  detail::to_scalar_impl(str, value, 10);
}

template <> void to_scalar(const char *str, unsigned long &value) {
  detail::to_scalar_impl(str, value, 10);
}

template <> void to_scalar(const char *str, long long &value) {
  detail::to_scalar_impl(str, value, 10);
}

template <> void to_scalar(const char *str, unsigned long long &value) {
  detail::to_scalar_impl(str, value, 10);
}
#endif // __cplusplus >= 201703L
#endif // __GNUC__ < 11

#if __cplusplus >= 201703L
template <typename T> T to_scalar(std::string_view str) {
  T value;
  to_scalar(str, value);
  return value;
}
#else
template <typename T> T to_scalar(const char *str) {
  T value;
  to_scalar(str, value);
  return value;
}
#endif // __cplusplus >= 201703L
} // namespace util
