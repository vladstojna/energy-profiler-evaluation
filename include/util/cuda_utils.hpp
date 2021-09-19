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
    void copy(host_buffer<T>& dest, const device_buffer<T>& src, std::size_t count);
    template<typename T>
    void copy(device_buffer<T>& dest, const host_buffer<T>& src, std::size_t count);
    template<typename T>
    void copy(device_buffer<T>& dest, const device_buffer<T>& src, std::size_t count);

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
        host_buffer(const device_buffer<element_type>& other) :
            host_buffer(other.size())
        {
            copy(*this, other, size());
        }

        host_buffer& operator=(const device_buffer<element_type>& other)
        {
            copy(*this, other, size());
            return *this;
        };

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
            copy(*this, other, size());
        }

        device_buffer& operator=(const device_buffer& other)
        {
            return *this = device_buffer(other);
        }

        device_buffer(host_buffer<element_type>&& other) = delete;
        device_buffer(const host_buffer<element_type>& other) :
            device_buffer(other.size())
        {
            copy(*this, other, size());
        }

        device_buffer& operator=(const host_buffer<element_type>& other)
        {
            copy(*this, other, size());
            return *this;
        };

        ~device_buffer() = default;
    };

    template<typename T>
    device_buffer(const host_buffer<T>&)->device_buffer<T>;

    template<typename T>
    void copy(host_buffer<T>& dest, const device_buffer<T>& src, std::size_t count)
    {
        copy_impl(dest.get(), src.get(), count, detail::device_to_host{});
    }

    template<typename T>
    void copy(device_buffer<T>& dest, const host_buffer<T>& src, std::size_t count)
    {
        copy_impl(dest.get(), src.get(), count, detail::host_to_device{});
    }

    template<typename T>
    void copy(device_buffer<T>& dest, const device_buffer<T>& src, std::size_t count)
    {
        copy_impl(dest.get(), src.get(), count, detail::device_to_device{});
    }
}