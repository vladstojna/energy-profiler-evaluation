#pragma once

#include "detail/unique_buffer.hpp"
#include "buffer.hpp"

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <iostream>

namespace util
{
    class device_exception : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    std::string get_cuda_error_str(std::string_view comment, cudaError_t err)
    {
        return std::string(comment).append(": ").append(cudaGetErrorString(err));
    }

    template<typename T>
    T* device_alloc(std::size_t count)
    {
        T* ptr;
        auto status = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T));
        if (status != cudaSuccess)
            throw device_exception(get_cuda_error_str("cudaMalloc error", status));
        return ptr;
    }

    template<typename T>
    void device_free(T* ptr)
    {
        auto status = cudaFree(ptr);
        if (status != cudaSuccess)
            std::cerr << "Error freeing device memory: " << cudaGetErrorString(status) << "\n";
    }

    namespace detail
    {
        template<typename InputIt>
        using is_random_access_iterator = std::is_convertible<
            typename std::iterator_traits<InputIt>::iterator_category,
            std::random_access_iterator_tag>;

        template<typename InputIt, typename T>
        using is_iterator_same_no_cv =
            std::is_same<T, typename std::iterator_traits<InputIt>::value_type>;

        template<typename InputIt, typename T>
        using is_iterator_compatible =
            std::conjunction<
            is_random_access_iterator<InputIt>,
            is_iterator_same_no_cv<InputIt, T>>;

        template<auto func>
        struct as_lambda : std::integral_constant<decltype(func), func>
        {};

        struct host_to_device {};
        struct device_to_host {};
        struct device_to_device {};

        template<typename T>
        void copy_impl(
            T* dest, const T* src, std::size_t count, cudaMemcpyKind kind, std::string_view msg)
        {
            auto res = cudaMemcpy(dest, src, count * sizeof(T), kind);
            if (res != cudaSuccess)
                throw device_exception(get_cuda_error_str(msg, res));
        }

        template<typename T>
        void copy_impl(T* dest, const T* src, std::size_t count, host_to_device)
        {
            detail::copy_impl(dest, src, count,
                cudaMemcpyHostToDevice,
                "cudaMemcpyHostToDevice error");
        }

        template<typename T>
        void copy_impl(T* dest, const T* src, std::size_t count, device_to_host)
        {
            detail::copy_impl(dest, src, count,
                cudaMemcpyDeviceToHost,
                "cudaMemcpyDeviceToHost error");
        }

        template<typename T>
        void copy_impl(T* dest, const T* src, std::size_t count, device_to_device)
        {
            detail::copy_impl(dest, src, count,
                cudaMemcpyDeviceToDevice,
                "cudaMemcpyDeviceToDevice error");
        }
    }

    template<typename T>
    class device_buffer :
        private detail::unique_buffer<T, detail::as_lambda<device_free<T>>>
    {
        using inherited = detail::unique_buffer<T, detail::as_lambda<device_free<T>>>;
    public:
        using size_type = typename inherited::size_type;
        using value_type = typename inherited::value_type;
        using pointer = typename inherited::pointer;
        using const_pointer = typename inherited::const_pointer;
        using reference = typename inherited::reference;
        using const_reference = typename inherited::const_reference;

        using inherited::get;
        using inherited::size;
        using inherited::operator bool;

        device_buffer() noexcept = default;

        explicit device_buffer(size_type size) :
            inherited(device_alloc<value_type>(size), size)
        {}

        device_buffer(device_buffer&& other) noexcept = default;
        device_buffer& operator=(device_buffer&& other) noexcept = default;

        device_buffer(const device_buffer& other) :
            device_buffer(other.size())
        {
            copy(other, size(), *this);
        }

        device_buffer& operator=(const device_buffer& other)
        {
            return *this = device_buffer(other);
        }

        template<typename InputIt,
            typename = std::enable_if_t<detail::is_iterator_compatible<InputIt, value_type>::value>
        > device_buffer(InputIt start, InputIt end) :
            device_buffer(std::distance(start, end))
        {
            copy(start, end, *this);
        }

        ~device_buffer() = default;
    };

    template<typename Iter>
    device_buffer(Iter, Iter)->device_buffer<typename std::iterator_traits<Iter>::value_type>;

    template<typename T>
    struct device_buffer_view
    {
        using value_type = T;
        using size_type = std::size_t;

        const value_type* data;
        size_type size;

        explicit device_buffer_view(const value_type* data, size_type size) noexcept :
            data{ data },
            size{ size }
        {}

        explicit device_buffer_view() noexcept :
            device_buffer_view{ nullptr, 0 }
        {}

        device_buffer_view(const device_buffer<value_type>& buffer) noexcept :
            device_buffer_view{ buffer.get(), buffer.size() }
        {}
    };

    template<typename InputIt, typename T,
        typename = std::enable_if_t<detail::is_iterator_compatible<InputIt, T>::value>
    > void copy(InputIt start, InputIt end, device_buffer<T>& into)
    {
        copy_impl(into.get(), &*start, std::distance(start, end), detail::host_to_device{});
    }

    template<typename InputIt, typename T,
        typename = std::enable_if_t<detail::is_iterator_compatible<InputIt, T>::value>
    > void copy(
        const device_buffer<T>& from,
        typename device_buffer<T>::size_type count,
        InputIt into)
    {
        copy_impl(&*into, from.get(), count, detail::device_to_host{});
    }

    template<typename T>
    void copy(const device_buffer_view<T>& from,
        typename device_buffer_view<T>::size_type count,
        device_buffer<T>& into)
    {
        copy_impl(into.get(), from.data, count, detail::device_to_device{});
    }

    template<typename T>
    void copy(const device_buffer_view<T>& from, device_buffer<T>& into)
    {
        copy(from, from.size, into);
    }

    template<typename T>
    void zero(device_buffer<T>& x)
    {
        auto res = cudaMemset(x.get(), 0, x.size() * sizeof(T));
        if (res != cudaSuccess)
            throw device_exception(get_cuda_error_str("cudaMemset error", res));
    }
}
