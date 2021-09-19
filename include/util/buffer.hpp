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
        using element_type = typename inherited::element_type;
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

        explicit buffer(size_type size) :
            inherited(new T[size], size)
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

        void swap(buffer& other)
        {
            inherited::swap(other);
        }
    };
}
