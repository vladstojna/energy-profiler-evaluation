#pragma once

#if __cplusplus >= 201703L
#include <charconv>
#include <string_view>
#endif // __cplusplus >= 201703L

#if __GNUC__ < 11 || __cplusplus < 201703L
#include <cstdlib>
#include <locale>
#endif // __GNUC__ < 11 || __cplusplus < 201703L

#if __cplusplus < 201703L
#include <numeric>
#include <type_traits>
#endif // __cplusplus < 201703L

namespace util
{
    // #if __cplusplus >= 201703L
    //     using string_view = std::string_view;
    // #else
    //     using string_view = const char*;
    // #endif

#if __cplusplus >= 201703L
    // std::from_chars is only available in C++17 and later
    template<typename T>
    void to_scalar(std::string_view str, T& value)
    {
        auto [dummy, ec] = std::from_chars(str.begin(), str.end(), value);
        (void)dummy;
        if (auto code = std::make_error_code(ec))
            throw std::system_error(code);
    }
#endif // __cplusplus >= 201703L

#if __GNUC__ < 11 || __cplusplus < 201703L
    // specialize floating point conversions if GCC version < 11
    // if C++ is older than 17 then implement conversion for all supported scalar types
    namespace detail
    {
    #if __cplusplus >= 201703L
        // In C++17 and later std::string_view is passed which is not guaranteed to be
        // null-terminated, thus a temporary copy is necessary since C standard
        // scalar conversion functions expect a null-terminated string
        template<typename Real, typename Func>
        void to_scalar_impl(std::string_view str, Real& value, Func func)
        {
            std::string tmp(str);
            std::locale loc;
            std::locale::global(std::locale::classic());
            errno = 0;
            Real val = func(tmp.c_str(), nullptr);
            if (errno)
            {
                std::locale::global(loc);
                throw std::system_error(
                    std::error_code(static_cast<int>(errno), std::generic_category()));
            }
            std::locale::global(loc);
            value = val;
        }
    #else
        // Implementation available in https://en.cppreference.com/w/cpp/types/conjunction
        template<typename...>
        struct conjunction : std::true_type {};
        template<typename T>
        struct conjunction<T> : T {};
        template<typename T, typename... Rest>
        struct conjunction<T, Rest...> :
            std::conditional_t<bool(T::value), conjunction<Rest...>, T>
        {};

        template<typename Scalar, typename Func, typename... Args>
        void to_scalar_impl(const char* str, Scalar& value, Func func, Args... args)
        {
            std::locale loc;
            std::locale::global(std::locale::classic());
            errno = 0;
            Scalar val = func(str, nullptr, args...);
            if (errno)
            {
                std::locale::global(loc);
                throw std::system_error(
                    std::error_code(static_cast<int>(errno), std::generic_category()));
            }
            std::locale::global(loc);
            value = val;
        }

        template<
            typename SmallerInt,
            typename BiggerInt,
            std::enable_if_t<conjunction<std::is_signed<SmallerInt>, std::is_signed<BiggerInt>>::value, bool> = true
        > bool is_in_range(BiggerInt x)
        {
            return x >= std::numeric_limits<SmallerInt>::min()
                && x <= std::numeric_limits<SmallerInt>::max();
        }

        template<
            typename SmallerInt,
            typename BiggerInt,
            std::enable_if_t<conjunction<std::is_unsigned<SmallerInt>, std::is_unsigned<BiggerInt>>::value, bool> = true
        > bool is_in_range(BiggerInt x)
        {
            return x <= std::numeric_limits<SmallerInt>::max();
        }

        template<typename BiggerInt, typename SmallerInt, typename Func, typename... Args>
        void to_scalar_impl_overflow(
            const char* str, SmallerInt& value, Func func, Args... args)
        {
            BiggerInt tmp;
            to_scalar_impl(str, tmp, func, args...);
            if (!is_in_range<SmallerInt>(tmp))
                throw std::system_error(
                    std::error_code(static_cast<int>(ERANGE), std::generic_category()));
            SmallerInt smaller = tmp;
            value = smaller;
        }
    #endif // __cplusplus >= 201703L
    }

#if __cplusplus >= 201703L
    template<>
    void to_scalar(std::string_view str, float& value)
    {
        detail::to_scalar_impl(str, value, std::strtof);
    }

    template<>
    void to_scalar(std::string_view str, double& value)
    {
        detail::to_scalar_impl(str, value, std::strtod);
    }

    template<>
    void to_scalar(std::string_view str, long double& value)
    {
        detail::to_scalar_impl(str, value, std::strtold);
    }
#else
    void to_scalar(const char* str, float& value)
    {
        detail::to_scalar_impl(str, value, std::strtof);
    }

    void to_scalar(const char* str, double& value)
    {
        detail::to_scalar_impl(str, value, std::strtod);
    }

    void to_scalar(const char* str, long double& value)
    {
        detail::to_scalar_impl(str, value, std::strtold);
    }

    void to_scalar(const char* str, char& value)
    {
        detail::to_scalar_impl_overflow<long>(
            str, value, std::strtol, 10);
    }

    void to_scalar(const char* str, short& value)
    {
        detail::to_scalar_impl_overflow<long>(
            str, value, std::strtol, 10);
    }

    void to_scalar(const char* str, int& value)
    {
        detail::to_scalar_impl_overflow<long>(
            str, value, std::strtol, 10);
    }

    void to_scalar(const char* str, long& value)
    {
        detail::to_scalar_impl(str, value, std::strtol, 10);
    }

    void to_scalar(const char* str, long long& value)
    {
        detail::to_scalar_impl(str, value, std::strtoll, 10);
    }

    void to_scalar(const char* str, unsigned char& value)
    {
        detail::to_scalar_impl_overflow<unsigned long>(
            str, value, std::strtoul, 10);
    }

    void to_scalar(const char* str, unsigned short& value)
    {
        detail::to_scalar_impl_overflow<unsigned long>(
            str, value, std::strtoul, 10);
    }

    void to_scalar(const char* str, unsigned int& value)
    {
        detail::to_scalar_impl_overflow<unsigned long>(
            str, value, std::strtoul, 10);
    }

    void to_scalar(const char* str, unsigned long& value)
    {
        detail::to_scalar_impl(str, value, std::strtoul, 10);
    }

    void to_scalar(const char* str, unsigned long long& value)
    {
        detail::to_scalar_impl(str, value, std::strtoull, 10);
    }
#endif // __cplusplus >= 201703L
#endif // __GNUC__ < 11

#if __cplusplus >= 201703L
    template<typename T>
    T to_scalar(std::string_view str)
    {
        T value;
        to_scalar(str, value);
        return value;
    }
#else
    template<typename T>
    T to_scalar(const char* str)
    {
        T value;
        to_scalar(str, value);
        return value;
    }
#endif // __cplusplus >= 201703L
}
