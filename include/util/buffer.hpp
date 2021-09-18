#pragma once

#include "detail/unique_buffer.hpp"

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

        using inherited::get;
        using inherited::size;
        using inherited::operator bool;
        using inherited::operator[];

        buffer(size_type size) :
            inherited(new T[size], size)
        {}

        void swap(buffer& other)
        {
            inherited::swap(other);
        }
    };
}
