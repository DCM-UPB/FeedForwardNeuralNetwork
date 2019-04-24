#ifndef QNETS_TEMPL_TEMPLLAYER_HPP
#define QNETS_TEMPL_TEMPLLAYER_HPP

#include "qnets/templ/DerivConfig.hpp"

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <type_traits>

namespace templ
{
// --- TemplNet Layers

// Layer Config
//
// To pass non-input layer configurations as variadic parameter pack
template <int N_OUT, class ACTFType>
struct LayerConfig
{
    static constexpr int noutput = N_OUT;
    using ACTF_Type = ACTFType;

    static constexpr int size() { return noutput; }
};


// The actual Layer class
//
template <typename ValueT, int NET_NINPUT, int NET_NOUTPUT, int NBETA_NEXT, int N_IN, int N_OUT, class ACTFType, DerivConfig DCONF>
class TemplLayer: public LayerConfig<N_OUT, ACTFType>
{
public:
    // N_IN dependent sizes
    static constexpr int ninput = N_IN;
    static constexpr int nbeta = (N_IN + 1)*N_OUT;
    static constexpr int net_nin = NET_NINPUT;
    static constexpr int net_nout = NET_NOUTPUT;

    // Sizes which also depend on DCONF
    static constexpr StaticDFlags<DCONF> dconf{};
    static constexpr int nd1 = dconf.d1 ? NET_NINPUT*N_OUT : 0; // number of input derivative values
    static constexpr int nd1_prev = dconf.d1 ? NET_NINPUT*N_IN : 0; // number of deriv values from previous layer
    static constexpr int nd2 = dconf.d2 ? nd1 : 0;
    static constexpr int nd2_prev = dconf.d2 ? nd1_prev : 0;

    static_assert(NBETA_NEXT % (1 + N_OUT) == 0, ""); // -> BUG!
    static constexpr int nout_next = NBETA_NEXT/(1 + N_OUT);
    static constexpr int nvd1 = dconf.vd1 ? NET_NOUTPUT*N_OUT : 0; // number of backprop grad values
    static constexpr int nvd1_next = dconf.vd1 ? NET_NOUTPUT*nout_next : 0; // number of backprop grad values from previous layer

private: // arrays
    std::array<ValueT, N_OUT> _out{};
    // the deriv arrays could be quite large, so we need to heap allocate
    const std::unique_ptr<std::array<ValueT, nd1>> _d1_ptr{std::make_unique<std::array<ValueT, nd1>>()};
    const std::unique_ptr<std::array<ValueT, nd2>> _d2_ptr{std::make_unique<std::array<ValueT, nd2>>()};
    const std::unique_ptr<std::array<ValueT, nvd1>> _vd1_ptr{std::make_unique<std::array<ValueT, nvd1>>()};

    std::array<ValueT, dconf.d1 || dconf.vd1 ? N_OUT : 0> _ad1{}; // activation function d1
    std::array<ValueT, dconf.d2 ? N_OUT : 0> _ad2{}; // activation function d2

public: // public member variables
    ACTFType actf{}; // the activation function
    std::array<ValueT, nbeta> beta{}; // the weights

    // public const output references
    constexpr const std::array<ValueT, N_OUT> &out() const { return _out; }
    constexpr const std::array<ValueT, nd1> &d1() const { return *_d1_ptr; }
    constexpr const std::array<ValueT, nd2> &d2() const { return *_d2_ptr; }
    constexpr const std::array<ValueT, nvd1> &vd1() const { return *_vd1_ptr; }

private:
    constexpr void _computeFeed(const ValueT input[])
    {
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1); // increments through the indices of the first non-offset beta per unit
            _out[i] = std::inner_product(input, input + ninput, beta.begin() + beta_i0, beta[beta_i0 - 1]/*bias weight*/); // found to be faster than loop
        }
    }

    constexpr void _computeActivation(bool flag_ad1, bool flag_ad2 /*is overriding*/)
    {
        if (flag_ad2) {
            actf.fd12(_out.begin(), _out.end(), _out.begin(), _ad1.begin(), _ad2.begin());
        }
        else if (flag_ad1) {
            actf.fd1(_out.begin(), _out.end(), _out.begin(), _ad1.begin());
        }
        else {
            actf.f(_out.begin(), _out.end(), _out.begin());
        }
    }

    constexpr void _computeOutput(const ValueT input[], DynamicDFlags dflags)
    {
        this->_computeFeed(input);
        this->_computeActivation(dflags.d1(), dflags.d2());
    }

    constexpr void _computeD1_Layer(const ValueT in_d1[])
    {
        auto &D1 = *_d1_ptr;
        D1.fill(0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1);
            const int d_i0 = i*NET_NINPUT;
            for (int j = 0; j < N_IN; ++j) {
                for (int k = 0; k < NET_NINPUT; ++k) {
                    D1[d_i0 + k] += beta[beta_i0 + j]*in_d1[j*NET_NINPUT + k];
                }
            }
            for (int l = d_i0; l < d_i0 + NET_NINPUT; ++l) {
                D1[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeD1_Input() // i.e. in_d1[i][i] = 1., else 0
    {
        auto &D1 = *_d1_ptr;
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(NET_NINPUT + 1);
            for (int j = 0; j < NET_NINPUT; ++j) {
                D1[i*NET_NINPUT + j] = _ad1[i]*beta[beta_i0 + j];
            }
        }
    }

    constexpr void _computeD12_Layer(const ValueT in_d1[], const ValueT in_d2[])
    {
        auto &D1 = *_d1_ptr;
        auto &D2 = *_d2_ptr;
        D1.fill(0.);
        D2.fill(0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1);
            const int d_i0 = i*NET_NINPUT;
            for (int j = 0; j < N_IN; ++j) {
                for (int k = 0; k < NET_NINPUT; ++k) {
                    D1[d_i0 + k] += beta[beta_i0 + j]*in_d1[j*NET_NINPUT + k];
                    D2[d_i0 + k] += beta[beta_i0 + j]*in_d2[j*NET_NINPUT + k];
                }
            }
            for (int l = d_i0; l < d_i0 + NET_NINPUT; ++l) {
                D2[l] = _ad1[i]*D2[l] + _ad2[i]*D1[l]*D1[l];
                D1[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeD12_Input()
    {
        auto &D1 = *_d1_ptr;
        auto &D2 = *_d2_ptr;
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(NET_NINPUT + 1);
            for (int j = 0; j < NET_NINPUT; ++j) {
                D1[i*NET_NINPUT + j] = _ad1[i]*beta[beta_i0 + j];
                D2[i*NET_NINPUT + j] = _ad2[i]*beta[beta_i0 + j]*beta[beta_i0 + j];
            }
        }
    }

    /*
    constexpr void _computeVD1_Layer(const ValueT input[], const ValueT in_vd1[])
    {
        auto &VD1 = *_vd1_ptr;
        VD1.fill(0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(N_IN + 1);
            const int d_i0 = i*ibeta_end;
            // add old elements
            for (int j = 0; j < N_IN; ++j) {
                for (int k = 0; k < IBETA_PREV_BEGIN; ++k) {
                    VD1[d_i0 + k] += beta[beta_i0 + j]*in_vd1[j*IBETA_BEGIN + k];
                }
                for (int k = IBETA_PREV_BEGIN + j*nblock_prev; k < IBETA_PREV_BEGIN + (j + 1)*nblock_prev; ++k) {
                    VD1[d_i0 + k] = beta[beta_i0 + j]*in_vd1[j*IBETA_BEGIN + k];
                }
            }
            // add new elements
            const int beta_inew = IBETA_BEGIN + beta_i0; // the first new non-offset beta index
            VD1[d_i0 + beta_inew - 1] = 1.; // the bias weight derivative
            std::copy(input, input + N_IN, VD1.begin() + d_i0 + beta_inew);
            for (int l = d_i0; l < d_i0 + ibeta_end; ++l) {
                VD1[l] *= _ad1[i];
            }
        }
    }

    constexpr void _computeVD1_Input(const ValueT input[])
    {
        auto &VD1 = *_vd1_ptr;
        VD1.fill(0.);
        for (int i = 0; i < N_OUT; ++i) {
            const int beta_i0 = 1 + i*(NET_NINPUT + 1);
            const int d_i0 = i*ibeta_end;
            VD1[d_i0 + beta_i0 - 1] = 1.; // the bias weight derivative
            std::copy(input, input + NET_NINPUT, VD1.begin() + d_i0 + beta_i0);
            for (int l = d_i0; l < d_i0 + ibeta_end; ++l) {
                VD1[l] *= _ad1[i];
            }
        }
    }*/

    constexpr void _forwardInput(const ValueT input[], DynamicDFlags dflags)
    {
        // statically secure this call (i.e. using it on non-input layer will not compile)
        static_assert(N_IN == NET_NINPUT, "[TemplLayer::ForwardInput] N_IN != NET_NINPUT");

        dflags = dflags.AND(dconf); // AND static and dynamic conf
        this->_computeOutput(input, dflags);

        // fill diagonal d1,d2
        if (dflags.d2()) {
            this->_computeD12_Input();
        }
        else if (dflags.d1()) {
            this->_computeD1_Input();
        }
    }


    constexpr void _forwardLayer(const ValueT input[], const ValueT in_d1[], const ValueT in_d2[], DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        this->_computeOutput(input, dflags);

        // input derivs
        if (dflags.d2()) {
            this->_computeD12_Layer(in_d1, in_d2);
        }
        else if (dflags.d1()) {
            this->_computeD1_Layer(in_d1);
        }
    }


    constexpr void _backwardOutput(DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        static_assert(N_OUT == NET_NOUTPUT, "[TemplLayer::BackwardOutput] N_OUT != NET_NOUTPUT");
        auto &VD1 = *_vd1_ptr;

        VD1.fill(0.);
        if (!dflags.vd1()) { return; }
            for (int i = 0; i < NET_NOUTPUT; ++i) {
                VD1[i*NET_NOUTPUT + i] = _ad1[i];
            }
    }

    constexpr void _backwardLayer(const ValueT vd1_next[], const ValueT beta_next[], DynamicDFlags dflags)
    {
        dflags = dflags.AND(dconf); // AND static and dynamic conf
        auto &VD1 = *_vd1_ptr;
        VD1.fill(0.);
        if (!dflags.vd1()) { return; }

        for (int i = 0; i < NET_NOUTPUT; ++i) {
            const int d_i0 = i*N_OUT;
            for (int j = 0; j < nout_next; ++j) {
                for (int k = 0; k < N_OUT; ++k) {
                    VD1[d_i0 + k] += beta_next[1 + j*(N_OUT + 1) + k] * vd1_next[i*nout_next + j];
                }
            }
            for (int k = 0; k < N_OUT; ++k) {
                VD1[d_i0 + k] *= _ad1[k];
            }
        }
    }

public: // public propagate methods
    // We support 3 different ways to provide arrays for propagate calls:
    // Array: Bounds statically checked due to type system
    // Vector: Runtime bounds checking (once per call)
    // C-Style (Pointer): No bounds checking


    // --- Propagation of input data (not layer)

    constexpr void ForwardInput(const std::array<ValueT, NET_NINPUT> &input, DynamicDFlags dflags)
    {
        _forwardInput(input.begin(), dflags);
    }

    constexpr void ForwardInput(const std::vector<ValueT> &input, DynamicDFlags dflags)
    {
        assert(input.size() == N_IN);
        _forwardInput(input.begin(), dflags);
    }

    constexpr void ForwardInput(const ValueT input[], DynamicDFlags dflags)
    {
        _forwardInput(input, dflags);
    }


    // --- Forward Propagation of layer data

    constexpr void ForwardLayer(const std::array<ValueT, N_IN> &input, const std::array<ValueT, nd1_prev> &in_d1, const std::array<ValueT, nd2_prev> &in_d2, DynamicDFlags dflags)
    {
        _forwardLayer(input.begin(), in_d1.begin(), in_d2.begin(), dflags);
    }

    constexpr void ForwardLayer(const std::vector<ValueT> &input, const std::vector<ValueT> &in_d1, const std::vector<ValueT> &in_d2, DynamicDFlags dflags)
    {
        assert(input.size() == N_IN);
        assert(in_d1.size() == nd1_prev);
        assert(in_d2.size() == nd2_prev);
        _forwardLayer(input.begin(), in_d1.begin(), in_d2.begin(), dflags);
    }

    constexpr void ForwardLayer(const ValueT input[], const ValueT in_d1[], const ValueT in_d2[], DynamicDFlags dflags)
    {
        _forwardLayer(input, in_d1, in_d2, dflags);
    }


    // --- Backward Propagation for an output layer

    constexpr void BackwardOutput(DynamicDFlags dflags)
    {
        _backwardOutput(dflags);
    }


    // --- Backward Propagation for a hidden layer

    constexpr void BackwardLayer(const std::array<ValueT, nvd1_next> &vd1_next, const std::array<ValueT, NBETA_NEXT> &beta_next, DynamicDFlags dflags)
    {
        _backwardLayer(vd1_next.begin(), beta_next.begin(), dflags);
    }

    constexpr void BackwardLayer(const std::vector<ValueT> &vd1_next, const std::vector<ValueT> &beta_next, DynamicDFlags dflags)
    {
        assert(vd1_next.size() == nvd1_next);
        assert(beta_next.size() == NBETA_NEXT);
        _backwardLayer(vd1_next.begin(), beta_next.begin(), dflags);
    }

    constexpr void BackwardLayer(const ValueT vd1_next[], const ValueT beta_next[], DynamicDFlags dflags)
    {
        _backwardLayer(vd1_next, beta_next, dflags);
    }
};

// --- propagateLayers helper to propagate input through a tuple of layers
//     The function uses the recursive subroutines below to iterate through the tuple.

namespace detail
{
// Recursive ForwardProp over tuple
template <class TupleT>
constexpr void fwdprop_layers_impl(TupleT &/*layers*/, DynamicDFlags /*dflags*/, std::index_sequence<>) {}

template <class TupleT, size_t I, size_t ... Is>
constexpr void fwdprop_layers_impl(TupleT &layers, DynamicDFlags dflags, std::index_sequence<I, Is...>)
{
    const auto &prev_layer = std::get<I>(layers);
    std::get<I + 1>(layers).ForwardLayer(prev_layer.out(), prev_layer.d1(), prev_layer.d2(), dflags);
    fwdprop_layers_impl<TupleT>(layers, dflags, std::index_sequence<Is...>{});
}

template <class TupleT>
constexpr void backprop_layers_impl(TupleT &/*layers*/, DynamicDFlags /*dflags*/, std::index_sequence<>) {}

template <class TupleT, size_t I, size_t ... Is>
constexpr void backprop_layers_impl(TupleT &layers, DynamicDFlags dflags, std::index_sequence<I, Is...>)
{
    constexpr size_t idx = sizeof...(Is);
    const auto &next_layer = std::get<idx + 1>(layers);
    std::get<idx>(layers).BackwardLayer(next_layer.vd1(), next_layer.beta, dflags);
    backprop_layers_impl<TupleT>(layers, dflags, std::index_sequence<Is...>{});
}

template <int ibeta_begin, int nbeta_net, class LayerT, class ArrayT1, class ArrayT2>
constexpr void calc_grad_layer(LayerT &layer, const ArrayT1 &input, ArrayT2 &vd1, DynamicDFlags dflags)
{
    if (!dflags.vd1()) {
        std::fill(vd1.begin(), vd1.end(), 0.);
        return;
    }
    const auto &LVD1 = layer.vd1();
    for (int i = 0; i < layer.net_nout; ++i) {
        const int lvd1_i0 = i*layer.noutput;
        for (int j = 0; j < layer.noutput; ++j) {
            const int gvd1_i0 = i*nbeta_net + ibeta_begin + j*(1 + layer.ninput) + 1; // global vd1 index of first non-offset weight
            vd1[gvd1_i0 - 1] = LVD1[lvd1_i0 + j]; // bias weight gradient
            for (int k = 0; k < layer.ninput; ++k) {
                vd1[gvd1_i0 + k] = input[k] * LVD1[lvd1_i0 + j];
            }
        }
    }
}

template <int ibeta_begin, int nbeta_net, class TupleT, class ArrayT1, class ArrayT2>
constexpr void grad_layers_impl(TupleT &/*layers*/, const ArrayT1 &/*input*/, ArrayT2 &/*vd1*/, DynamicDFlags /*dflags*/, std::index_sequence<>) {}

template <int ibeta_begin, int nbeta_net, class TupleT, class ArrayT1, class ArrayT2, size_t I, size_t ... Is>
constexpr void grad_layers_impl(TupleT &layers, const ArrayT1 &input, ArrayT2 &vd1, DynamicDFlags dflags, std::index_sequence<I, Is...>)
{
    const auto &this_layer = std::get<I>(layers);

    calc_grad_layer<ibeta_begin, nbeta_net>(this_layer, input, vd1, dflags);
    grad_layers_impl<ibeta_begin + this_layer.nbeta, nbeta_net, TupleT>(layers, this_layer.out(), vd1, dflags, std::index_sequence<Is...>{});
}
} // detail

// The public function
// ArrayT1/2 can be any random-access std container
// vd1 should consist of net_noutput blocks of size nbeta_net
template <int nbeta, class TupleT, class ArrayT1, class ArrayT2>
constexpr void propagateLayers(TupleT &layers, const ArrayT1 &input, ArrayT2 &vd1, DynamicDFlags dflags)
{
    using namespace detail;
    constexpr int nlayer = static_cast<int>(std::tuple_size<TupleT>::value);

    // fdwprop
    std::get<0>(layers).ForwardInput(input, dflags);
    fwdprop_layers_impl<TupleT>(layers, dflags, std::make_index_sequence<nlayer - 1>{});

    // backprop
    std::get<nlayer - 1>(layers).BackwardOutput(dflags);
    backprop_layers_impl<TupleT>(layers, dflags, std::make_index_sequence<nlayer - 1>{});

    // store backprop grads into vd1
    grad_layers_impl<0, nbeta, TupleT>(layers, input, vd1, dflags, std::make_index_sequence<nlayer>{});
}
} // templ

#endif