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
    class device_buffer;
    template<typename T>
    class host_buffer;

    template<typename T>
    void copy(
        typename host_buffer<T>::const_iterator start,
        typename host_buffer<T>::const_iterator end,
        device_buffer<T>& into);
    template<typename T>
    void copy(const device_buffer<T>& from, typename host_buffer<T>::iterator into);
    template<typename T>
    void copy(const device_buffer<T>& from, host_buffer<T>& into);
    template<typename T>
    void copy(const host_buffer<T>& from, device_buffer<T>& into);
    template<typename T>
    void copy(const device_buffer<T>& from, device_buffer<T>& into);

    template<typename T>
    class host_buffer : private buffer<T>
    {
        using inherited = buffer<T>;
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

        explicit host_buffer(size_type size) :
            inherited(size)
        {}

        host_buffer(device_buffer<element_type>&& other) = delete;
        explicit host_buffer(const device_buffer<element_type>& other) :
            host_buffer(other.size())
        {
            copy(other, *this);
        }

        host_buffer(inherited&& other) :
            inherited(std::move(other))
        {}

        host_buffer(const inherited& other) :
            inherited(other)
        {}
    };

    template<typename T>
    host_buffer(const device_buffer<T>&)->host_buffer<T>;

    template<typename T>
    class device_buffer :
        private detail::unique_buffer<T, detail::as_lambda<device_free<T>>>
    {
        using inherited = detail::unique_buffer<T, detail::as_lambda<device_free<T>>>;
    public:
        using size_type = typename inherited::size_type;
        using element_type = typename inherited::element_type;

        using inherited::get;
        using inherited::size;
        using inherited::operator bool;

        explicit device_buffer(size_type size) :
            inherited(device_alloc<element_type>(size), size)
        {}

        device_buffer(device_buffer&& other) noexcept = default;
        device_buffer& operator=(device_buffer&& other) noexcept = default;

        device_buffer(const device_buffer& other) :
            device_buffer(other.size())
        {
            copy(other, *this);
        }

        device_buffer& operator=(const device_buffer& other)
        {
            return *this = device_buffer(other);
        }

        device_buffer(
            typename host_buffer<element_type>::const_iterator start,
            typename host_buffer<element_type>::const_iterator end)
            :
            device_buffer(std::distance(start, end))
        {
            copy(start, end, *this);
        }

        device_buffer(
            typename host_buffer<element_type>::const_iterator start,
            typename host_buffer<element_type>::size_type count)
            :
            device_buffer(start, start + count)
        {}

        device_buffer(host_buffer<element_type>&& other) = delete;
        explicit device_buffer(const host_buffer<element_type>& other) :
            device_buffer(other.begin(), other.end())
        {}

        ~device_buffer() = default;
    };

    template<typename T>
    device_buffer(const host_buffer<T>&)->device_buffer<T>;

    template<typename Iter>
    device_buffer(Iter, Iter)->device_buffer<typename std::iterator_traits<Iter>::value_type>;

    template<typename Iter>
    device_buffer(
        Iter,
        typename host_buffer<typename std::iterator_traits<Iter>::value_type>::size_type
    )->device_buffer<typename std::iterator_traits<Iter>::value_type>;

    template<typename T>
    void copy(
        typename host_buffer<T>::const_iterator start,
        typename host_buffer<T>::const_iterator end,
        device_buffer<T>& into)
    {
        copy_impl(into.get(), &*start, std::distance(start, end), detail::host_to_device{});
    }

    template<typename T>
    void copy(const device_buffer<T>& from, typename host_buffer<T>::iterator into)
    {
        copy_impl(&*into, from.get(), from.size(), detail::device_to_host{});
    }

    template<typename T>
    void copy(const device_buffer<T>& from, host_buffer<T>& into)
    {
        copy(from, into.begin());
    }

    template<typename T>
    void copy(const host_buffer<T>& from, device_buffer<T>& into)
    {
        copy(from.begin(), from.end(), into);
    }

    template<typename T>
    void copy(const device_buffer<T>& from, device_buffer<T>& into)
    {
        copy_impl(into.get(), from.get(), from.size(), detail::device_to_device{});
    }
}