#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <iostream>

namespace tp
{
    struct period_data
    {
        std::chrono::nanoseconds interval;
        std::size_t initial_size = 0;
    };

    class printer
    {
    public:
        struct impl;

        printer(std::ostream & = std::cout);
        printer(const std::optional<period_data>&, std::ostream & = std::cout);

        ~printer();

        printer(const printer&) = delete;
        printer(printer&&) = delete;
        printer& operator=(const printer&) = delete;
        printer& operator=(printer&&) = delete;

    private:
        std::unique_ptr<impl> _impl;
    };
}
