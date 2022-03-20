/**
 * @file This file is part of EDGE.
 *
 * @author Alexander Breuer (anbreuer AT ucsd.edu)
 *
 * @section LICENSE
 * Copyright (c) 2019, Alexander Breuer
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *this list of conditions and the following disclaimer in the documentation
 *and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *may be used to endorse or promote products derived from this software without
 *specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 * Data structures for volume integrations.
 **/
#ifndef EDGE_DG_VOL_INT_HPP
#define EDGE_DG_VOL_INT_HPP

#include "Basis.h"
#include "constants.hpp"
#include "data/Dynamic.h"

namespace edge {
namespace dg {
template <t_entityType TL_T_EL, unsigned short TL_O_SP>
class VolInt;
}
}  // namespace edge

/**
 * Functions for the setup of volume integration matrices.
 *
 * @paramt TL_T_EL element type.
 * @paramt TL_O_SP spatial order.
 **/
template <t_entityType TL_T_EL, unsigned short TL_O_SP>
class edge::dg::VolInt {
 private:
  //! number of dimensions
  static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

  //! number of DG modes
  static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES(TL_T_EL, TL_O_SP);
  //! even padded number of element modes
  static unsigned short const TL_N_MDS_PAD = EVEN_PAD(TL_N_MDS);

  /**
   * Stores the stiffness matrices as dense.
   *
   * @param i_stiff dense stiffness matrices.
   * @param io_dynMem dynamic memory management, which will be used for the
   *respective allocations.
   * @param o_stiff will contain pointers to memory for the individual matrices.
   *
   * @paramt TL_T_REAL real type.
   **/
  // add 0 in B. dst must be filed with zeros and of size (K+1)(N)

  template <typename T>
  static void vnni_swap(T const *i_src, T *io_dest, size_t K, size_t M) {
    for (size_t l_k = 0; l_k < K; l_k += 2) {
      for (size_t l_m = 0; l_m < M; l_m++) {
        io_dest[2 * l_m + l_k * M] = i_src[l_m + l_k * M];
        io_dest[2 * l_m + l_k * M + 1] = i_src[l_m + (l_k + 1) * M];
      }
    }

    if (K % 2 == 1) {
      for (size_t l_m = 0; l_m < M; l_m++) {
        io_dest[2 * l_m + (K - 1) * M] = i_src[l_m + (K - 1) * M];
        io_dest[2 * l_m + (K - 1) * M + 1] = 0;
      }
    }
  }

  template <typename TL_T_REAL>
  static void storeStiffDense(
      TL_T_REAL const i_stiff[TL_N_DIS][TL_N_MDS][TL_N_MDS],
      data::Dynamic &io_dynMem, TL_T_REAL *o_stiff[TL_N_DIS]) {
    /*
#if defined(PP_FP32_BF16_APPROX)
std::size_t l_size = TL_N_DIS * std::size_t(TL_N_MDS) * TL_N_MDS_PAD;
l_size *= sizeof(TL_T_REAL);
TL_T_REAL *l_stiffRaw =
    (TL_T_REAL *)io_dynMem.allocate(l_size, 4096, false, true);

// copy data
for (unsigned short l_di = 0; l_di < TL_N_DIS; l_di++) {
  vnni_swap(&i_stiff[l_di][0][0],
            &l_stiffRaw[l_di * std::size_t(TL_N_MDS) * TL_N_MDS_PAD],
            TL_N_MDS, TL_N_MDS);
}

// assign pointers
for (unsigned short l_di = 0; l_di < TL_N_DIS; l_di++) {
  o_stiff[l_di] = l_stiffRaw + l_di * std::size_t(TL_N_MDS) * TL_N_MDS_PAD;
}
*/
    //#else

    // allocate raw memory for the stiffness matrices
    std::size_t l_size = TL_N_DIS * std::size_t(TL_N_MDS) * TL_N_MDS;
    l_size *= sizeof(TL_T_REAL);
    TL_T_REAL *l_stiffRaw =
        (TL_T_REAL *)io_dynMem.allocate(l_size, 4096, false, true);

    // copy data
    std::size_t l_en = 0;
    for (unsigned short l_di = 0; l_di < TL_N_DIS; l_di++) {
      for (unsigned short l_m0 = 0; l_m0 < TL_N_MDS; l_m0++) {
        for (unsigned short l_m1 = 0; l_m1 < TL_N_MDS; l_m1++) {
          l_stiffRaw[l_en] = i_stiff[l_di][l_m0][l_m1];
          l_en++;
        }
      }
    }

    // assign pointers
    for (unsigned short l_di = 0; l_di < TL_N_DIS; l_di++) {
      o_stiff[l_di] = l_stiffRaw + l_di * std::size_t(TL_N_MDS) * TL_N_MDS;
    }
    //#endif
  }

 public:
  /**
   * Stores the stiffness matrices as dense.
   *
   * @param io_dynMem dynamic memory management, which will be used for the
   *respective allocations.
   * @param o_stiff will contain pointers to memory for the individual matrices.
   *
   * @paramt TL_T_REAL real type.
   **/
  template <typename TL_T_REAL>
  static void storeStiffDense(data::Dynamic &io_dynMem,
                              TL_T_REAL *o_stiff[TL_N_DIS]) {
    // formulation of the basis in terms of the reference element
    dg::Basis l_basis(TL_T_EL, TL_O_SP);

    // get stiffness matrices
    TL_T_REAL l_stiff[TL_N_DIS][TL_N_MDS][TL_N_MDS];
    l_basis.getStiffMm1Dense(TL_N_MDS, l_stiff[0][0], false);

    // store
    storeStiffDense(l_stiff, io_dynMem, o_stiff);
  }
};
#endif