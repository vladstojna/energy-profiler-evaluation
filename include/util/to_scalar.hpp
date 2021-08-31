#pragma once

#include <charconv>
#include <string_view>

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

    template<typename T>
    T to_scalar(std::string_view str)
    {
        T value;
        to_scalar(str, value);
        return value;
    }
}
