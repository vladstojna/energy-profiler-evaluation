#include <timeprinter/printer.hpp>
#include <util/to_scalar.hpp>

#include <charconv>
#include <thread>

namespace
{
    tp::printer g_tpr;

    __attribute__((noinline)) void sleep(const std::chrono::milliseconds& dur)
    {
        std::this_thread::sleep_for(dur);
    }

    void print_usage(const char* prog)
    {
        std::cerr << "Usage: " << prog << " <ms>\n";
    }
}

int main(int argc, char** argv)
{
    try
    {
        if (argc < 2)
        {
            print_usage(argv[0]);
            throw std::invalid_argument("Not enough arguments");
        }
        auto ms = util::to_scalar<std::chrono::milliseconds::rep>(argv[1]);
        tp::sampler s(g_tpr);
        sleep(std::chrono::milliseconds(ms));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}
