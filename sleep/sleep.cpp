#include <timeprinter/printer.hpp>

#include <charconv>
#include <thread>

namespace
{
    __attribute__((noinline)) void sleep(const std::chrono::milliseconds& dur)
    {
        std::this_thread::sleep_for(dur);
    }

    void print_usage(const char* prog)
    {
        std::cerr << "Usage: " << prog << " <ms>\n";
    }

    template<typename T>
    void to_scalar(std::string_view str, T& value)
    {
        auto [dummy, ec] = std::from_chars(str.begin(), str.end(), value);
        (void)dummy;
        if (auto code = std::make_error_code(ec))
            throw std::system_error(code);
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
        std::chrono::milliseconds::rep ms;
        to_scalar(argv[1], ms);
        tp::printer tpr;
        sleep(std::chrono::milliseconds(ms));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}
