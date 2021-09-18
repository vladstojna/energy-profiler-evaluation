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
    template<typename T>
    using host_buffer = buffer<T>;

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

        template<typename T>
        void device_copy_impl(
            T* dest, const T* src, std::size_t count, cudaMemcpyKind kind, std::string_view msg)
        {
            auto res = cudaMemcpy(dest, src, count * sizeof(T), kind);
            if (res != cudaSuccess)
                throw device_exception(get_cuda_error_str(msg, res));
        }
    }

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

        device_buffer(size_type size) :
            inherited(device_alloc<T>(size), size)
        {}

        void swap(device_buffer& other)
        {
            inherited::swap(other);
        }
    };

    template<typename T>
    void copy_to_device(T* dest, const T* src, std::size_t count)
    {
        detail::device_copy_impl(dest, src, count, cudaMemcpyHostToDevice,
            "cudaMemcpyHostToDevice error");
    }

    template<typename T>
    void copy_from_device(T* dest, const T* src, std::size_t count)
    {
        detail::device_copy_impl(dest, src, count, cudaMemcpyDeviceToHost,
            "cudaMemcpyDeviceToHost error");
    }

    template<typename T>
    void copy(host_buffer<T>& dest, const device_buffer<T>& src, std::size_t count)
    {
        copy_from_device(dest.get(), src.get(), count);
    }

    template<typename T>
    void copy(device_buffer<T>& dest, const host_buffer<T>& src, std::size_t count)
    {
        copy_to_device(dest.get(), src.get(), count);
    }
}