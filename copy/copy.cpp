#include <util/to_scalar.hpp>
#include <util/cuda_utils.hpp>
#include <timeprinter/printer.hpp>

#include <cassert>

#define NO_INLINE __attribute__((noinline))

namespace
{
    tp::printer g_tpr;

    enum class copy_direction
    {
        to_device,
        from_device,
    };

    struct arguments
    {
        copy_direction direction;
        std::size_t size = 0;
        std::size_t iters = 1;

        arguments(int argc, const char* const* argv)
        {
            if (argc < 3)
            {
                usage(argv[0]);
                throw std::invalid_argument("Not enough arguments");
            }
            direction = get_direction(lowercase(argv[1]));
            util::to_scalar(argv[2], size);
            assert_positive(size, "size");
            if (argc > 3)
            {
                util::to_scalar(argv[3], iters);
                assert_positive(iters, "iters");
            }
        }

    private:
        std::string lowercase(const char* str)
        {
            std::string lower(str);
            std::transform(lower.begin(), lower.end(), lower.begin(),
                [](unsigned char c)
                {
                    return std::tolower(c);
                });
            return lower;
        }

        copy_direction get_direction(const std::string& x)
        {
            if (x == "to")
                return copy_direction::to_device;
            if (x == "from")
                return copy_direction::from_device;
            throw std::invalid_argument("direction must be 'to' or 'from'");
        }

        void assert_positive(std::size_t x, std::string name)
        {
            assert(x);
            if (!x)
                throw std::invalid_argument(std::move(name.append(" must be greater than 0")));
        }

        void usage(const char* prog)
        {
            std::cerr << "Usage: " << prog << " {to,from} <size in bytes> <iters>\n";
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
