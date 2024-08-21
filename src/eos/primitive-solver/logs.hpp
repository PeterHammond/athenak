#ifndef EOS_PRIMITIVE_SOLVER_LOGS_HPP_
#define EOS_PRIMITIVE_SOLVER_LOGS_HPP_
//========================================================================================
// Not-Quite-Transcendental log and exp functions
// Reproduced from https://github.com/lanl/not-quite-transcendental/tree/main
//========================================================================================
/*
© (or copyright) 2022. Triad National Security, LLC. All rights
reserved.  This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S.  Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the
U.S. Department of Energy/National Nuclear Security
Administration. The Government is granted for itself and others acting
on its behalf a nonexclusive, paid-up, irrevocable worldwide license
in this material to reproduce, prepare derivative works, distribute
copies to the public, perform publicly and display publicly, and to
permit others to do so.

This program is open source under the BSD-3 License.  Redistribution
and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#include <limits>

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"

namespace Primitive{

class LogPolicy {
  public:
  LogPolicy() = default;
  ~LogPolicy() = default;

  // KOKKOS_INLINE_FUNCTION Real log2(const Real x) const {return std::numeric_limits<Real>::quiet_NaN();}
  // KOKKOS_INLINE_FUNCTION Real exp2(const Real x) const {return std::numeric_limits<Real>::quiet_NaN();}
};

class NormalLogs : public LogPolicy {
  public:
    NormalLogs() = default;
    ~NormalLogs() = default;

    KOKKOS_INLINE_FUNCTION Real log2_(const Real x) const {
      return Kokkos::log2(x);
    }

    KOKKOS_INLINE_FUNCTION Real exp2_(const Real x) const {
      return Kokkos::exp2(x);
    }

};

class NQTLogs : public LogPolicy {
  public:
    NQTLogs() = default;
    ~NQTLogs() = default;

    KOKKOS_INLINE_FUNCTION Real log2_(const Real x) const {
      // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
      // these are floating point numbers as reinterpreted as integers.
      // as_int(1.0)
      constexpr int64_t one_as_int = 4607182418800017408;
      // 1./static_cast<double>(as_int(2.0) - as_int(1.0))
      constexpr Real scale_down = 2.22044604925031e-16;
      return static_cast<Real>(as_int(x) - one_as_int) * scale_down;
    }

    KOKKOS_INLINE_FUNCTION Real exp2_(const Real x) const {
      // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
      // these are floating point numbers as reinterpreted as integers.
      // as_int(1.0)
      constexpr int64_t one_as_int = 4607182418800017408;
      // as_int(2.0) - as_int(1.0)
      constexpr Real scale_up = 4503599627370496;
      return as_double(static_cast<int64_t>(x*scale_up) + one_as_int);
    }
  private:
    KOKKOS_INLINE_FUNCTION int64_t as_int(const Real f) const {
      Real f_ = f;
      return *reinterpret_cast<int64_t*>(&f_);
    }

    KOKKOS_INLINE_FUNCTION Real as_double(const int64_t i) const {
      int64_t i_ = i;
      return *reinterpret_cast<Real*>(&i_);
    }
};

} // namespace Primitive

/*
KOKKOS_INLINE_FUNCTION int64_t as_int(Real f) {
  return *reinterpret_cast<int64_t*>(&f);
}

KOKKOS_INLINE_FUNCTION Real as_double(int64_t i) {
  return *reinterpret_cast<Real*>(&i);
}

KOKKOS_INLINE_FUNCTION Real NQT_log2(const Real x) {
  // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
  // these are floating point numbers as reinterpreted as integers.
  // as_int(1.0)
  constexpr int64_t one_as_int = 4607182418800017408;
  // 1./static_cast<double>(as_int(2.0) - as_int(1.0))
  constexpr Real scale_down = 2.22044604925031e-16;
  return static_cast<Real>(as_int(x) - one_as_int) * scale_down;
}

KOKKOS_INLINE_FUNCTION Real NQT_exp2(const Real x) {
  // Magic numbers constexpr because C++ doesn't constexpr reinterpret casts
  // these are floating point numbers as reinterpreted as integers.
  // as_int(1.0)
  constexpr int64_t one_as_int = 4607182418800017408;
  // as_int(2.0) - as_int(1.0)
  constexpr Real scale_up = 4503599627370496;
  return as_double(static_cast<int64_t>(x*scale_up) + one_as_int);
}

KOKKOS_INLINE_FUNCTION Real log2(Real x, bool use_NQT=false) {
  if (use_NQT) {
    return NQT_log2(x);
  } else {
    return std::log2(x);
  }
}

KOKKOS_INLINE_FUNCTION Real exp2(Real x, bool use_NQT=false) {
  if (use_NQT) {
    return NQT_exp2(x);
  } else {
    return std::exp2(x);
  }
}
*/

#endif