#pragma once

#include <memory>
#include <type_traits>

namespace util {
template <typename T, typename Deleter> class unique_handle {
public:
  using handle_type = T;
  using value_type = std::remove_pointer_t<handle_type>;
  using deleter_type = Deleter;

  unique_handle() noexcept = default;

  explicit unique_handle(handle_type handle) noexcept : _handle(handle) {}

  operator handle_type() noexcept { return _handle.get(); }

  explicit operator bool() const noexcept { return bool(_handle); }

private:
  template <typename, typename = void> struct is_complete : std::false_type {};

  template <typename U>
  struct is_complete<U, std::void_t<decltype(sizeof(U))>> : std::true_type {};

  constexpr static const bool is_valid_type =
      std::is_pointer_v<handle_type> && !is_complete<value_type>::value;
  static_assert(is_valid_type, "T must be a pointer to an incomplete type");

  std::unique_ptr<value_type, deleter_type> _handle;
};
} // namespace util
