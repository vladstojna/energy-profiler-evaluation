#pragma once

#include "detail/unique_buffer.hpp"

#include <algorithm>

namespace util
{
    template<typename T>
    class buffer : private detail::unique_buffer<T>
    {
    private:
        using inherited = detail::unique_buffer<T>;

    public:
        using size_type = typename inherited::size_type;
        using value_type = typename inherited::value_type;
        using pointer = typename inherited::pointer;
        using const_pointer = typename inherited::const_pointer;
        using reference = typename inherited::reference;
        using const_reference = typename inherited::const_reference;
        using iterator = typename inherited::iterator;
        using const_iterator = typename inherited::const_iterator;

        using inherited::get;
        using inherited::size;
        using inherited::operator bool;
        using inherited::operator[];
        using inherited::begin;
        using inherited::end;
        using inherited::cbegin;
        using inherited::cend;

        buffer() noexcept = default;

        explicit buffer(size_type size) :
            inherited(new value_type[size], size)
        {}

        buffer(buffer&& other) noexcept = default;
        buffer& operator=(buffer&& other) noexcept = default;

        buffer(const buffer& other) :
            buffer(other.size())
        {
            std::copy(std::begin(other), std::end(other), std::begin(*this));
        }

        buffer& operator=(const buffer& other)
        {
            return *this = buffer(other);
        }

        ~buffer() = default;
    };
}
