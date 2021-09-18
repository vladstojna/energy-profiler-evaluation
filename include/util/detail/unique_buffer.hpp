#pragma once

#include <memory>

namespace util
{
    namespace detail
    {
        template<typename T, typename Deleter = std::default_delete<T[]>>
        class unique_buffer
        {
        private:
            using holder = std::unique_ptr<T[], Deleter>;

        public:
            using pointer = typename holder::pointer;
            using element_type = typename holder::element_type;
            using deleter_type = typename holder::deleter_type;
            using size_type = std::size_t;

            template<typename U>
            unique_buffer(U p, size_type size) noexcept :
                _data(p),
                _size(size)
            {}

            pointer get() const noexcept
            {
                return _data.get();
            }

            size_type size() const noexcept
            {
                return _size;
            }

            explicit operator bool() const noexcept
            {
                return bool(_data);
            }

            std::add_lvalue_reference_t<element_type> operator[](size_type i) const
            {
                return _data[i];
            }

            void swap(unique_buffer& other) noexcept
            {
                _data.swap(other._data);
                _size = std::exchange(other._size, _size);
            }

        private:
            holder _data;
            size_type _size;
        };
    }
}
