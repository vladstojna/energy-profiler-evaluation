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

            template<bool Const>
            struct iterator_impl
            {
                using iterator_category = std::random_access_iterator_tag;
                using difference_type = std::ptrdiff_t;
                using size_type = std::size_t;
                using value_type = element_type;
                using pointer = std::conditional_t<Const, const value_type*, value_type*>;
                using reference = std::conditional_t<Const, const value_type&, value_type&>;

                iterator_impl(pointer ptr) :
                    _ptr(ptr)
                {}

                template<bool OtherConst, bool ThisConst = Const,
                    std::enable_if_t<!OtherConst && ThisConst, bool> = true
                > iterator_impl(iterator_impl<OtherConst> iter) :
                    iterator_impl(&*iter)
                {}

                template<bool CConst = Const, std::enable_if_t<!CConst, bool> = true>
                reference operator*() { return *_ptr; }

                template<bool CConst = Const, std::enable_if_t<CConst, bool> = true>
                reference operator*() const { return *_ptr; }

                template<bool CConst = Const, std::enable_if_t<!CConst, bool> = true>
                pointer operator->() { return _ptr; }

                template<bool CConst = Const, std::enable_if_t<CConst, bool> = true>
                pointer operator->() const { return _ptr; }

                template<bool CConst = Const, std::enable_if_t<!CConst, bool> = true>
                reference operator[](size_type i) { return _ptr[i]; }

                template<bool CConst = Const, std::enable_if_t<CConst, bool> = true>
                reference operator[](size_type i) const { return _ptr[i]; }

                iterator_impl& operator++()
                {
                    _ptr++;
                    return *this;
                }

                iterator_impl operator++(int)
                {
                    iterator_impl current = *this;
                    ++(*this);
                    return current;
                }

                iterator_impl& operator--()
                {
                    _ptr--;
                    return *this;
                }

                iterator_impl operator--(int)
                {
                    iterator_impl current = *this;
                    --(*this);
                    return current;
                }

                iterator_impl& operator+=(size_type val)
                {
                    _ptr += val;
                    return *this;
                }

                iterator_impl& operator-=(size_type val)
                {
                    _ptr -= val;
                    return *this;
                }

                friend iterator_impl operator+(const iterator_impl it, size_type val)
                {
                    return it._ptr + val;
                }

                friend iterator_impl operator+(size_type val, const iterator_impl it)
                {
                    return it + val;
                }

                friend iterator_impl operator-(const iterator_impl it, size_type val)
                {
                    return it._ptr - val;
                }

                friend difference_type operator-(const iterator_impl lhs, const iterator_impl rhs)
                {
                    return lhs._ptr - rhs._ptr;
                }

                friend bool operator==(const iterator_impl lhs, const iterator_impl rhs)
                {
                    return lhs._ptr == rhs._ptr;
                }

                friend bool operator!=(const iterator_impl lhs, const iterator_impl rhs)
                {
                    return !(lhs == rhs);
                }

                friend bool operator<(const iterator_impl lhs, const iterator_impl rhs)
                {
                    return lhs._ptr < rhs._ptr;
                }

                friend bool operator>(const iterator_impl lhs, const iterator_impl rhs)
                {
                    return lhs._ptr > rhs._ptr;
                }

                friend bool operator<=(const iterator_impl lhs, const iterator_impl rhs)
                {
                    return lhs._ptr <= rhs._ptr;
                }

                friend bool operator>=(const iterator_impl lhs, const iterator_impl rhs)
                {
                    return lhs._ptr >= rhs._ptr;
                }

            private:
                pointer _ptr;
            };

            using iterator = iterator_impl<false>;
            using const_iterator = iterator_impl<true>;

            template<typename U>
            unique_buffer(U p, size_type size) noexcept :
                _data(p),
                _size(size)
            {}

            pointer get() const noexcept { return _data.get(); }
            size_type size() const noexcept { return _size; }
            explicit operator bool() const noexcept { return bool(_data); }

            std::add_lvalue_reference_t<element_type> operator[](size_type i) const
            {
                return _data[i];
            }

            iterator begin() noexcept { return iterator(_data.get()); }
            iterator end() noexcept { return iterator(_data.get() + _size); }
            const_iterator begin() const noexcept { return const_iterator(_data.get()); }
            const_iterator end() const noexcept { return const_iterator(_data.get() + _size); }
            const_iterator cbegin() const noexcept { return begin(); }
            const_iterator cend() const noexcept { return end(); }

        private:
            holder _data;
            size_type _size;
        };
    }
}
