#ifndef ASSERT_H
#define ASSERT_H
#include <cassert>
#include <iostream>
inline constexpr void NOT_USED()
{
}

template<class T, class... ARGS>
inline constexpr void NOT_USED(T&& first, ARGS&&... args)
{
#pragma warning(push)
#pragma warning(disable : 4101)
    static_cast<void>(first);
#pragma warning(pop)
    NOT_USED(args...);
}

template<typename... ARGS>
void HIP_ASSERT(bool test_result, ARGS&&... args)
{
#ifdef _DEBUG
    if (!test_result)
    {
        (std::cerr << ... << args) << std::endl;
        abort();
    }
#else
    NOT_USED(test_result, args...);
#endif
}

#endif // ASSERT