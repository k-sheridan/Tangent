#pragma once

#include <tuple>

namespace ArgMin {

template <typename... T>
struct VariableGroup {};

template <typename... T>
struct ErrorTermGroup {};

template <int T>
struct Dimension {};

template <typename T>
struct Scalar {};

template <typename T>
struct LossFunction {};

template <typename T>
struct TypedIndex {
    size_t index;
};

namespace internal {
    // Used to get the index of a tuple.
    template <class T, class Tuple>
    struct Index;

    template <class T, class... Types>
    struct Index<T, std::tuple<T, Types...>> {
        static const std::size_t value = 0;
    };

    template <class T, class U, class... Types>
    struct Index<T, std::tuple<U, Types...>> {
        static const std::size_t value = 1 + Index<T, std::tuple<Types...>>::value;
    };

    // static_for implementation.
    template <class Tup, class Func, std::size_t ...Is>
    constexpr void static_for_impl(Tup&& t, Func &&f, std::index_sequence<Is...> )
    {
       ( f(std::integral_constant<std::size_t, Is>{}, std::get<Is>(t)),... );
    }

    template <class ... T, class Func >
    constexpr void static_for(std::tuple<T...>&t, Func &&f)
    {
       static_for_impl(t, std::forward<Func>(f), std::make_index_sequence<sizeof...(T)>{});
    }
    /* USAGE EXAMPLE
    // args: (tuple index, tuple value)
    int main()
    {
        auto t = std::make_tuple( 1, 22, 3, 4 );

        std::size_t weighted = 0;
        static_for(t, [&] (auto i, auto w) { weighted += (i+1) * w; });

        std::cout << "Weighted: " << weighted << std::endl;

        return 0;
    }*/


    // Test if type is in tuple
    template <typename V, typename T>
    struct Is_in_tuple;

    template <typename V, typename T0, typename... T>
    struct Is_in_tuple <V, std::tuple<T0, T...> >
    {
        static const bool value = Is_in_tuple<V, std::tuple<T...> >::value;
    };

    template <typename V, typename... T>
    struct Is_in_tuple <V, std::tuple<V, T...> >
    {
        static const bool value = true;
    };

    template <typename V>
    struct Is_in_tuple <V, std::tuple<> >
    {
        static const bool value = false;
    };

}

}