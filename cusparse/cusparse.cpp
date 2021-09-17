#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>

#include <algorithm>
#include <cassert>
#include <random>

namespace
{
    tp::printer g_tpr;

    template<typename T>
    class device_buffer
    {
    public:
        using value_type = T;

    public:
        device_buffer(std::size_t size) :
            _ptr(nullptr),
            _size(size)
        {
            auto res = cudaMalloc(reinterpret_cast<void**>(&_ptr), size * sizeof(value_type));
            if (res != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(res));
        }

        ~device_buffer()
        {
            auto res = cudaFree(_ptr);
            if (res != cudaSuccess)
                std::cerr << "Error freeing device memory: " << cudaGetErrorString(res) << "\n";
        }

        const T* get() const
        {
            return _ptr;
        }

        T* get()
        {
            return _ptr;
        }

        operator const T* () const
        {
            return get();
        }

        operator T* ()
        {
            return get();
        }

        std::size_t size() const
        {
            return _size;
        }

    private:
        T* _ptr;
        std::size_t _size;
    };

    template<typename T, typename Deleter>
    class opaque_handle
    {
        T handle;
        Deleter del;

    public:
        template<typename Creator, typename... Args>
        opaque_handle(Deleter d, Creator c, Args&&... args) :
            handle(c(std::forward<Args>(args)...)),
            del(d)
        {}

        ~opaque_handle()
        {
            del(handle);
        }

        operator T()
        {
            return handle;
        }
    };

    std::string get_cuda_error(std::string_view comment, cudaError_t err)
    {
        return std::string(comment).append(": ").append(cudaGetErrorString(err));
    }

    std::string get_cusparse_error(std::string_view comment, cusparseStatus_t err)
    {
        return std::string(comment).append(": ").append(cusparseGetErrorString(err));
    }

    auto cusparse_create = []()
    {
        cusparseHandle_t handle;
        cusparseStatus_t status = cusparseCreate(&handle);
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(get_cusparse_error("Error creating cuSPARSE", status));
        return handle;
    };

    auto cusparse_destroy = [](cusparseHandle_t handle)
    {
        cusparseStatus_t status = cusparseDestroy(handle);
        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "Error destroying cuSPARSE: "
                << cusparseGetErrorString(status) << "\n";
        }
    };

    auto cusparse_spgemm_descr_create = []()
    {
        cusparseSpGEMMDescr_t descriptor;
        cusparseStatus_t status = cusparseSpGEMM_createDescr(&descriptor);
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                get_cusparse_error("Error creating cusparseSpGEMMDescr_t", status));
        return descriptor;
    };

    auto cusparse_spgemm_descr_destroy = [](cusparseSpGEMMDescr_t descriptor)
    {
        cusparseStatus_t status = cusparseSpGEMM_destroyDescr(descriptor);
        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "Error destroying cusparseSpGEMMDescr_t: "
                << cusparseGetErrorString(status) << "\n";
        }
    };

    auto cusparse_csr_create = [](
        std::int32_t M, std::int32_t N, std::size_t NNZ,
        std::int32_t* rows_csr,
        std::int32_t* cols,
        auto&& values,
        cudaDataType_t data_type)
    {
        cusparseSpMatDescr_t mat;
        auto status = cusparseCreateCsr(
            &mat, M, N, NNZ, rows_csr, cols, values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, data_type);
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                get_cusparse_error("Error creating CSR matrix descriptor", status));
        return mat;
    };

    auto cusparse_csr_destroy = [](cusparseSpMatDescr_t desc)
    {
        auto status = cusparseDestroySpMat(desc);
        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "Error destroying cusparseSpMatDescr_t: "
                << cusparseGetErrorString(status) << "\n";
        }
    };

    using cusparse_lib_handle = opaque_handle<
        cusparseHandle_t,
        decltype(cusparse_destroy)
    >;
    using cusparse_spgemm_descr = opaque_handle<
        cusparseSpGEMMDescr_t,
        decltype(cusparse_spgemm_descr_destroy)
    >;
    using cusparse_csr_mat_descr = opaque_handle<
        cusparseSpMatDescr_t,
        decltype(cusparse_csr_destroy)
    >;

    namespace detail
    {
        template<typename T, typename = void>
        struct has_enum_value : std::false_type
        {};

        template<typename T>
        struct has_enum_value<T, std::void_t<decltype(T::enum_value)>> : std::true_type
        {};

        template<typename Real>
        struct data_type
        {};

        template<>
        struct data_type<float>
        {
            constexpr static const auto enum_value = CUDA_R_32F;
        };

        template<>
        struct data_type<double>
        {
            constexpr static const auto enum_value = CUDA_R_64F;
        };

        std::vector<std::int32_t> get_indices(
            std::int32_t N, std::size_t NNZ, std::mt19937_64& engine)
        {
            std::uniform_int_distribution<std::int32_t> dist{ 0, N };
            std::vector<std::int32_t> indices(NNZ);
            std::generate(indices.begin(), indices.end(), [&]() { return dist(engine); });
            return indices;
        }

        template<typename T>
        void copy_impl(
            T* dest, const T* src, std::size_t count, cudaMemcpyKind kind, std::string_view msg)
        {
            auto res = cudaMemcpy(dest, src, count * sizeof(T), kind);
            if (res != cudaSuccess)
                throw std::runtime_error(get_cuda_error(msg, res));
        }

        template<typename T>
        void copy_to_device(T* dest, const T* src, std::size_t count)
        {
            copy_impl(dest, src, count, cudaMemcpyHostToDevice, "Error copying host -> device");
        }

        template<typename T>
        void copy_from_device(T* dest, const T* src, std::size_t count)
        {
            copy_impl(dest, src, count, cudaMemcpyDeviceToHost, "Error copying device -> host");
        }

        void coo2csr(
            device_buffer<std::int32_t>& dest,
            std::int32_t rows,
            std::size_t nnz,
            cusparseHandle_t handle,
            std::mt19937_64& engine)
        {
            auto row_idxs = get_indices(rows, nnz, engine);
            std::sort(row_idxs.begin(), row_idxs.end());
            device_buffer<std::int32_t> drows(row_idxs.size());
            copy_to_device(drows.get(), row_idxs.data(), row_idxs.size());
            auto status = cusparseXcoo2csr(handle,
                drows,
                nnz,
                rows,
                dest,
                CUSPARSE_INDEX_BASE_ZERO);
            cudaDeviceSynchronize();
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(get_cusparse_error("Error converting COO to CSR", status));
        }


        template<typename Real>
        void spgemm_impl(
            std::int32_t M,
            std::int32_t N,
            std::int32_t K,
            std::size_t A_nnz,
            std::size_t B_nnz,
            cusparseHandle_t handle,
            std::mt19937_64& engine)
        {
            static_assert(has_enum_value<data_type<Real>>::value, "Unsupported Real type");
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            const auto A_op_type = CUSPARSE_OPERATION_NON_TRANSPOSE;
            const auto B_op_type = CUSPARSE_OPERATION_NON_TRANSPOSE;
            const auto alg = CUSPARSE_SPGEMM_DEFAULT;
            const auto dtype = data_type<Real>::enum_value;

            tp::sampler smp(g_tpr);
            std::vector<Real> a_values(A_nnz);
            std::vector<Real> b_values(B_nnz);
            auto a_cols = get_indices(K, A_nnz, engine);
            auto b_cols = get_indices(N, B_nnz, engine);
            std::generate(a_values.begin(), a_values.end(), gen);
            std::generate(b_values.begin(), b_values.end(), gen);

            smp.do_sample();

            device_buffer<Real> da_values(a_values.size());
            device_buffer<std::int32_t> da_cols(a_cols.size());
            device_buffer<std::int32_t> da_rows_csr(M + 1);
            device_buffer<Real> db_values(b_values.size());
            device_buffer<std::int32_t> db_cols(b_cols.size());
            device_buffer<std::int32_t> db_rows_csr(K + 1);
            device_buffer<std::int32_t> dc_rows_csr(M + 1);

            coo2csr(da_rows_csr, M, A_nnz, handle, engine);
            coo2csr(db_rows_csr, K, B_nnz, handle, engine);
            copy_to_device(da_values.get(), a_values.data(), a_values.size());
            copy_to_device(da_cols.get(), a_cols.data(), a_cols.size());
            copy_to_device(db_values.get(), b_values.data(), b_values.size());
            copy_to_device(db_cols.get(), b_cols.data(), b_cols.size());

            smp.do_sample();

            cusparse_spgemm_descr spgemm_desc(
                cusparse_spgemm_descr_destroy,
                cusparse_spgemm_descr_create);
            cusparse_csr_mat_descr A(
                cusparse_csr_destroy,
                cusparse_csr_create,
                M, K, A_nnz, da_rows_csr, da_cols, da_values, dtype);
            cusparse_csr_mat_descr B(
                cusparse_csr_destroy,
                cusparse_csr_create,
                K, N, B_nnz, db_rows_csr, db_cols, db_values, dtype);
            cusparse_csr_mat_descr C(
                cusparse_csr_destroy,
                cusparse_csr_create,
                M, N, 0, nullptr, nullptr, nullptr, dtype);

            smp.do_sample();

            Real alpha = 1.0;
            Real beta = 0.0;
            std::size_t buffer_size1;
            auto status = cusparseSpGEMM_workEstimation(
                handle,
                A_op_type,
                B_op_type,
                &alpha, A, B, &beta, C,
                dtype,
                alg,
                spgemm_desc,
                &buffer_size1,
                nullptr);
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(
                    get_cusparse_error("1st cusparseSpGEMM_workEstimation error", status));
            device_buffer<std::uint8_t> dbuffer1(buffer_size1);
            status = cusparseSpGEMM_workEstimation(
                handle,
                A_op_type,
                B_op_type,
                &alpha, A, B, &beta, C,
                dtype,
                alg,
                spgemm_desc,
                &buffer_size1,
                dbuffer1);
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(
                    get_cusparse_error("2nd cusparseSpGEMM_workEstimation error", status));

            cudaDeviceSynchronize();
            smp.do_sample();

            std::size_t buffer_size2;
            status = cusparseSpGEMM_compute(
                handle,
                A_op_type,
                B_op_type,
                &alpha, A, B, &beta, C,
                dtype,
                alg,
                spgemm_desc,
                &buffer_size2,
                nullptr);
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(
                    get_cusparse_error("1st cusparseSpGEMM_compute error", status));
            device_buffer<std::uint8_t> dbuffer2(buffer_size2);
            status = cusparseSpGEMM_compute(
                handle,
                A_op_type,
                B_op_type,
                &alpha, A, B, &beta, C,
                dtype,
                alg,
                spgemm_desc,
                &buffer_size2,
                dbuffer2);
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(
                    get_cusparse_error("2nd cusparseSpGEMM_compute error", status));

            cudaDeviceSynchronize();
            smp.do_sample();

            std::int64_t C_nnz;
            {
                std::int64_t C_num_rows, C_num_cols;
                status = cusparseSpMatGetSize(C, &C_num_rows, &C_num_cols, &C_nnz);
                if (status != CUSPARSE_STATUS_SUCCESS)
                    throw std::runtime_error("Error getting matrix C size");
            }
            device_buffer<Real> dc_values(C_nnz);
            device_buffer<std::int32_t> dc_cols(C_nnz);
            status = cusparseCsrSetPointers(C, dc_rows_csr.get(), dc_cols.get(), dc_values.get());
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error("Error setting matrix C pointers");

            status = cusparseSpGEMM_copy(
                handle,
                A_op_type,
                B_op_type,
                &alpha, A, B, &beta, C,
                dtype,
                alg,
                spgemm_desc);

            std::vector<Real> c_values(C_nnz);
            std::vector<std::int32_t> c_rows_csr(M + 1);
            std::vector<std::int32_t> c_cols(C_nnz);
            copy_from_device(c_values.data(), dc_values.get(), c_values.size());
            copy_from_device(c_rows_csr.data(), dc_rows_csr.get(), c_rows_csr.size());
            copy_from_device(c_cols.data(), dc_cols.get(), c_cols.size());
        }
    }

    __attribute__((noinline)) void spdgemm(
        std::int32_t M,
        std::int32_t N,
        std::int32_t K,
        std::size_t A_nnz,
        std::size_t B_nnz,
        cusparseHandle_t handle,
        std::mt19937_64& engine)
    {
        detail::spgemm_impl<double>(M, N, K, A_nnz, B_nnz, handle, engine);
    }

    __attribute__((noinline)) void spsgemm(
        std::int32_t M,
        std::int32_t N,
        std::int32_t K,
        std::size_t A_nnz,
        std::size_t B_nnz,
        cusparseHandle_t handle,
        std::mt19937_64& engine)
    {
        detail::spgemm_impl<float>(M, N, K, A_nnz, B_nnz, handle, engine);
    }

    using work_func = void(*)(
        std::int32_t,
        std::int32_t,
        std::int32_t,
        std::size_t,
        std::size_t,
        cusparseHandle_t,
        std::mt19937_64&);

    struct cmdargs
    {
        std::int32_t m = 0;
        std::int32_t n = 0;
        std::int32_t k = 0;
        std::size_t a_nnz = 0;
        std::size_t b_nnz = 0;
        work_func func = nullptr;

        cmdargs(int argc, const char* const* argv)
        {
            if (argc < 7)
            {
                usage(argv[0]);
                throw std::invalid_argument("Not enough arguments");
            }
            std::string op_type = argv[1];
            std::transform(op_type.begin(), op_type.end(), op_type.begin(),
                [](unsigned char c) { return std::tolower(c); });

            util::to_scalar(argv[2], m);
            if (m < 0)
                throw std::invalid_argument("<m> must be positive");
            util::to_scalar(argv[3], n);
            if (n < 0)
                throw std::invalid_argument("<n> must be positive");
            util::to_scalar(argv[4], k);
            if (k < 0)
                throw std::invalid_argument("<k> must be positive");
            util::to_scalar(argv[5], a_nnz);
            util::to_scalar(argv[6], b_nnz);

            if (op_type == "dgemm")
                func = spdgemm;
            else if (op_type == "sgemm")
                func = spsgemm;
            else
            {
                usage(argv[0]);
                throw std::invalid_argument(std::string("invalid work type: ").append(argv[1]));
            }
            assert(func);
        }

        void do_work(cusparseHandle_t handle, std::mt19937_64& engine) const
        {
            func(m, n, k, a_nnz, b_nnz, handle, engine);
        }

    private:
        void usage(const char* prog)
        {
            std::cerr << "Usage: " << prog << " {dgemm,sgemm} <m> <n> <k> <A nnz> <B nnz>\n";
        }
    };
}

int main(int argc, char** argv)
{
    try
    {
        const cmdargs args(argc, argv);
        std::random_device rnd_dev;
        std::mt19937_64 engine{ rnd_dev() };
        cusparse_lib_handle handle(cusparse_destroy, cusparse_create);
        args.do_work(handle, engine);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}
