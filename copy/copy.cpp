#include <util/to_scalar.hpp>
#include <util/cuda_utils.hpp>
#include <timeprinter/printer.hpp>

#include <cassert>

#define NO_INLINE __attribute__((noinline))

namespace
{
    tp::printer g_tpr;

    struct arguments
    {
        std::size_t size = 0;
        std::size_t iters = 1;

        arguments(int argc, const char* const* argv)
        {
            if (argc < 2)
            {
                usage(argv[0]);
                throw std::invalid_argument("Not enough arguments");
            }
            util::to_scalar(argv[1], size);
            assert(size > 0);
            if (!size)
                throw std::invalid_argument("size must be greater than 0");
            if (argc > 2)
            {
                util::to_scalar(argv[2], iters);
                assert(iters >= 1);
                if (iters < 1)
                    throw std::invalid_argument("iters must be greater than 1");
            }
        }

    private:
        void usage(const char* prog)
        {
            std::cerr << "Usage: " << prog << " [size in bytes] {iters}\n";
        }
    };

    NO_INLINE void copy(
        const util::buffer<std::uint8_t>& from,
        util::device_buffer<std::uint8_t>& into,
        std::size_t iters)
    {
        tp::sampler smp(g_tpr);
        for (std::size_t i = 0; i < iters; i++)
        {
            util::copy(std::begin(from), std::end(from), into);
        }
    }

    void do_work(std::size_t n, std::size_t iters)
    {
        tp::sampler smp(g_tpr);
        util::buffer<std::uint8_t> buff(n);
        util::device_buffer<std::uint8_t> dev_buff(buff.size());
        copy(buff, dev_buff, iters);
    }
}

int main(int argc, char** argv)
{
    try
    {
        const arguments args{ argc, argv };
        do_work(args.size, args.iters);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
