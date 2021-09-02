#pragma once

#include <charconv>
#include <string_view>

#if __GNUC__ < 11
#include <cstdlib>
#include <locale>
#endif

namespace util
{
    template<typename T>
    void to_scalar(std::string_view str, T& value)
    {
        auto [dummy, ec] = std::from_chars(str.begin(), str.end(), value);
        (void)dummy;
        if (auto code = std::make_error_code(ec))
            throw std::system_error(code);
    }

    // specialize floating point conversions if GCC version < 11
#if __GNUC__ < 11
    namespace detail
    {
        template<typename Real, typename Func>
        void to_scalar_impl(std::string_view str, Real& value, Func func)
        {
            auto loc = std::setlocale(0, nullptr);
            if (!loc)
                throw std::runtime_error("Error querying locale");
            if (!std::setlocale(LC_ALL, "C"))
                throw std::runtime_error("Error setting locale");
            std::string tmp(str);
            errno = 0;
            Real val = func(tmp.c_str(), nullptr);
            if (errno)
            {
                if (!std::setlocale(LC_ALL, loc))
                    throw std::runtime_error("Error setting locale");
                throw std::system_error(
                    std::error_code(static_cast<int>(errno), std::generic_category()));
            }
            if (!std::setlocale(LC_ALL, loc))
                throw std::runtime_error("Error setting locale");
            value = val;
        }
    }

    template<>
    void to_scalar(std::string_view str, float& value)
    {
        detail::to_scalar_impl<float>(str, value, std::strtof);
    }

    template<>
    void to_scalar(std::string_view str, double& value)
    {
        detail::to_scalar_impl<double>(str, value, std::strtod);
    }

    template<>
    void to_scalar(std::string_view str, long double& value)
    {
        detail::to_scalar_impl<long double>(str, value, std::strtold);
    }
#endif // __GNUC__ < 11

    template<typename T>
    T to_scalar(std::string_view str)
    {
        T value;
        to_scalar(str, value);
        return value;
    }
}
