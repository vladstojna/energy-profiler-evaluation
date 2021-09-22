#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>
#include <util/cuda_utils.hpp>
#include <util/unique_handle.hpp>

#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <random>
#include <tuple>
#include <vector>

namespace
{
    tp::printer g_tpr;

    template<auto Func>
    struct func_obj : std::integral_constant<decltype(Func), Func> {};

    std::string get_cusparse_error(std::string_view comment, cusparseStatus_t err)
    {
        return std::string(comment).append(": ").append(cusparseGetErrorString(err));
    }

    cusparseHandle_t cusparse_create()
    {
        cusparseHandle_t handle;
        cusparseStatus_t status = cusparseCreate(&handle);
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(get_cusparse_error("Error creating cuSPARSE", status));
        return handle;
    }

    void cusparse_destroy(cusparseHandle_t handle)
    {
        cusparseStatus_t status = cusparseDestroy(handle);
        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "Error destroying cuSPARSE: "
                << cusparseGetErrorString(status) << "\n";
        }
    };

    using cusparse_handle = util::unique_handle<
        cusparseHandle_t,
        func_obj<cusparse_destroy>>;

    namespace detail
    {
        template<typename>
        struct cuda_data_type {};

        template<>
        struct cuda_data_type<float>
            : std::integral_constant<decltype(CUDA_R_32F), CUDA_R_32F>
        {};

        template<>
        struct cuda_data_type<double>
            : std::integral_constant<decltype(CUDA_R_64F), CUDA_R_64F>
        {};

        cusparseSpGEMMDescr_t cusparse_spgemm_descr_create()
        {
            cusparseSpGEMMDescr_t descriptor;
            cusparseStatus_t status = cusparseSpGEMM_createDescr(&descriptor);
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(
                    get_cusparse_error("Error creating cusparseSpGEMMDescr_t", status));
            return descriptor;
        };

        void cusparse_spgemm_descr_destroy(cusparseSpGEMMDescr_t descriptor)
        {
            cusparseStatus_t status = cusparseSpGEMM_destroyDescr(descriptor);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                std::cerr << "Error destroying cusparseSpGEMMDescr_t: "
                    << cusparseGetErrorString(status) << "\n";
            }
        };

        template<typename Real>
        cusparseSpMatDescr_t cusparse_csr_create(
            std::int32_t M, std::int32_t N,
            util::device_buffer<std::int32_t>& rows_csr,
            util::device_buffer<std::int32_t>& cols,
            util::device_buffer<Real>& values)
        {
            cusparseSpMatDescr_t mat;
            auto status = cusparseCreateCsr(
                &mat, M, N, cols.size(), rows_csr.get(), cols.get(), values.get(),
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                cuda_data_type<Real>::value);
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(
                    get_cusparse_error("Error creating CSR matrix descriptor", status));
            return mat;
        };

        template<typename Real>
        cusparseSpMatDescr_t cusparse_csr_create(std::int32_t M, std::int32_t N)
        {
            cusparseSpMatDescr_t mat;
            auto status = cusparseCreateCsr(
                &mat, M, N, 0, nullptr, nullptr, nullptr,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                cuda_data_type<Real>::value);
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(
                    get_cusparse_error("Error creating CSR matrix descriptor", status));
            return mat;
        };

        void cusparse_csr_destroy(cusparseSpMatDescr_t desc)
        {
            auto status = cusparseDestroySpMat(desc);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                std::cerr << "Error destroying cusparseSpMatDescr_t: "
                    << cusparseGetErrorString(status) << "\n";
            }
        };

        using cusparse_spgemm_descr = util::unique_handle<
            cusparseSpGEMMDescr_t,
            func_obj<cusparse_spgemm_descr_destroy>
        >;
        using cusparse_csr_mat_descr = util::unique_handle<
            cusparseSpMatDescr_t,
            func_obj<cusparse_csr_destroy>
        >;

        util::host_buffer<std::int32_t>
            get_coo_rows(std::int32_t M, std::int32_t N, std::size_t NNZ, std::mt19937_64& engine)
        {
            std::uniform_int_distribution<decltype(M)> dist{ 0, M - 1 };
            util::host_buffer<std::int32_t> indices{ NNZ };

            // generate random numbers but ensure
            // that there are no more than N equal values
            std::vector<std::int32_t> existing(M, 0);
            auto gen = [&]()
            {
                constexpr const std::int32_t max_iters = 100;
                for (auto i = 0; i < max_iters; i++)
                {
                    decltype(dist)::result_type val = dist(engine);
                    if (existing[val] < N)
                    {
                        existing[val]++;
                        return val;
                    }
                }
                std::cerr << "Could not generate unique random number after "
                    << max_iters << " tries, looking for first row number less than N\n";
                auto it = std::find_if(existing.begin(), existing.end(),
                    [N](std::int32_t val) { return val < N; });
                assert(it != existing.end());
                if (it == existing.end())
                    throw std::runtime_error("More non-zero entries than matrix entries");
                (*it)++;
                return static_cast<decltype(M)>(std::distance(existing.begin(), it));
            };

            std::generate(indices.begin(), indices.end(), gen);
            std::sort(indices.begin(), indices.end());
            return indices;
        }

        util::host_buffer<std::int32_t> get_coo_cols(
            std::int32_t N,
            std::mt19937_64& engine,
            const util::host_buffer<std::int32_t>& coo_rows)
        {
            assert(coo_rows.size());
            std::uniform_int_distribution<decltype(N)> dist{ 0, N - 1 };
            util::host_buffer<std::int32_t> indices{ coo_rows.size() };

            // sort the column indices for each row
            // assume coo_rows is already sorted
            for (auto [rprev, rit, cprev, cit] =
                std::tuple{
                    coo_rows.begin(),
                    std::next(coo_rows.begin()),
                    indices.begin(),
                    std::next(indices.begin())
                };
                rit <= coo_rows.end() && cit <= indices.end();
                rit++, cit++)
            {
                if (rit == coo_rows.end() || *rit > *rprev)
                {
                    // generate random numbers for this interval
                    // but numbers must be unique because (row, column) index pairs are unique
                    // in COO format
                    std::vector<bool> existing(N, false);
                    auto gen = [&]()
                    {
                        constexpr const std::int32_t max_iters = 100;
                        for (auto i = 0; i < max_iters; i++)
                        {
                            decltype(dist)::result_type val = dist(engine);
                            if (!existing[val])
                            {
                                existing[val] = true;
                                return val;
                            }
                        }
                        std::cerr << "Could not generate unique random number after "
                            << max_iters << " tries, looking for first row number less than N\n";
                        auto it = std::find(existing.begin(), existing.end(), false);
                        assert(it != existing.end());
                        if (it == existing.end())
                            throw std::runtime_error("More non-zero entries than matrix entries");
                        *it = true;
                        return static_cast<decltype(N)>(std::distance(existing.begin(), it));
                    };
                    std::generate(cprev, cit, gen);
                    std::sort(cprev, cit);
                    rprev = rit;
                    cprev = cit;
                }
            }
            return indices;
        }

        void coo2csr(
            cusparse_handle& handle,
            const util::host_buffer<std::int32_t>& coo,
            util::device_buffer<std::int32_t>& csr)
        {
            const util::device_buffer<std::int32_t> dcoo{ coo };
            auto status = cusparseXcoo2csr(
                handle,
                dcoo.get(),
                dcoo.size(),
                csr.size() - 1,
                csr.get(),
                CUSPARSE_INDEX_BASE_ZERO);
            cudaDeviceSynchronize();
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(get_cusparse_error("Error converting COO to CSR", status));
        }

        template<typename Real>
        typename util::host_buffer<Real>::size_type spgemm_impl(
            std::int32_t M,
            std::int32_t N,
            std::int32_t K,
            std::size_t A_nnz,
            std::size_t B_nnz,
            cusparse_handle& handle,
            std::mt19937_64& engine)
        {
            std::uniform_real_distribution<Real> dist{ 0.0, 1.0 };
            auto gen = [&]() { return dist(engine); };

            static constexpr const auto A_op_type = CUSPARSE_OPERATION_NON_TRANSPOSE;
            static constexpr const auto B_op_type = CUSPARSE_OPERATION_NON_TRANSPOSE;
            static constexpr const auto alg = CUSPARSE_SPGEMM_DEFAULT;
            static constexpr const auto dtype = cuda_data_type<Real>::value;

            tp::sampler smp(g_tpr);
            util::host_buffer<Real> a_values{ A_nnz };
            util::host_buffer<Real> b_values{ B_nnz };
            std::generate(a_values.begin(), a_values.end(), gen);
            std::generate(b_values.begin(), b_values.end(), gen);

            util::host_buffer<std::int32_t> a_coo_rows = get_coo_rows(M, K, A_nnz, engine);
            util::host_buffer<std::int32_t> a_coo_cols = get_coo_cols(K, engine, a_coo_rows);
            util::host_buffer<std::int32_t> b_coo_rows = get_coo_rows(K, N, B_nnz, engine);
            util::host_buffer<std::int32_t> b_coo_cols = get_coo_cols(N, engine, b_coo_rows);

            smp.do_sample();

            util::device_buffer da_values{ a_values };
            util::device_buffer da_coo_cols{ a_coo_cols };
            util::device_buffer db_values{ b_values };
            util::device_buffer db_coo_cols{ b_coo_cols };

            util::device_buffer<std::int32_t> da_csr_rows{ static_cast<std::size_t>(M) + 1 };
            util::device_buffer<std::int32_t> db_csr_rows{ static_cast<std::size_t>(K) + 1 };
            util::device_buffer<std::int32_t> dc_csr_rows{ static_cast<std::size_t>(M) + 1 };

            coo2csr(handle, a_coo_rows, da_csr_rows);
            coo2csr(handle, b_coo_rows, db_csr_rows);

            smp.do_sample();

            cusparse_spgemm_descr spgemm_desc(cusparse_spgemm_descr_create());
            cusparse_csr_mat_descr A(
                cusparse_csr_create(M, K, da_csr_rows, da_coo_cols, da_values));
            cusparse_csr_mat_descr B(
                cusparse_csr_create(K, N, db_csr_rows, db_coo_cols, db_values));
            cusparse_csr_mat_descr C(cusparse_csr_create<Real>(M, N));

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
            util::device_buffer<std::uint8_t> dbuffer1{ buffer_size1 };
            status = cusparseSpGEMM_workEstimation(
                handle,
                A_op_type,
                B_op_type,
                &alpha, A, B, &beta, C,
                dtype,
                alg,
                spgemm_desc,
                &buffer_size1,
                dbuffer1.get());
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
            util::device_buffer<std::uint8_t> dbuffer2{ buffer_size2 };
            status = cusparseSpGEMM_compute(
                handle,
                A_op_type,
                B_op_type,
                &alpha, A, B, &beta, C,
                dtype,
                alg,
                spgemm_desc,
                &buffer_size2,
                dbuffer2.get());
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
                assert(C_nnz >= 0);
                if (C_nnz < 0)
                    throw std::runtime_error("Matrix C non-zero entries < 0");
            }
            util::device_buffer<Real> dc_values{ static_cast<std::size_t>(C_nnz) };
            util::device_buffer<std::int32_t> dc_coo_cols{ static_cast<std::size_t>(C_nnz) };
            status = cusparseCsrSetPointers(
                C, dc_csr_rows.get(), dc_coo_cols.get(), dc_values.get());
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

            util::host_buffer c_values{ dc_values };
            util::host_buffer c_csr_rows{ dc_csr_rows };
            util::host_buffer c_coo_cols{ dc_coo_cols };
            return c_values.size();
        }
    }

    __attribute__((noinline)) auto spdgemm(
        std::int32_t M,
        std::int32_t N,
        std::int32_t K,
        std::size_t A_nnz,
        std::size_t B_nnz,
        cusparse_handle& handle,
        std::mt19937_64& engine)
    {
        return detail::spgemm_impl<double>(M, N, K, A_nnz, B_nnz, handle, engine);
    }

    __attribute__((noinline)) auto spsgemm(
        std::int32_t M,
        std::int32_t N,
        std::int32_t K,
        std::size_t A_nnz,
        std::size_t B_nnz,
        cusparse_handle& handle,
        std::mt19937_64& engine)
    {
        return detail::spgemm_impl<float>(M, N, K, A_nnz, B_nnz, handle, engine);
    }

    struct cmdargs
    {
        using work_func = decltype(&spdgemm);

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

        auto do_work(cusparse_handle& handle, std::mt19937_64& engine) const
        {
            return func(m, n, k, a_nnz, b_nnz, handle, engine);
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
        cusparse_handle handle(cusparse_create());
        auto nnz = args.do_work(handle, engine);
        std::cerr << "Resulting matrix: "
            << args.m << "x" << args.n
            << " with " << nnz << " non-zero entries\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}
