/**
 * @file This file is part of EDGE.
 *
 * @author Alexander Breuer (anbreuer AT ucsd.edu)
 *         Alexander Heinecke (alexander.heinecke AT intel.com)
 *
 * @section LICENSE
 * Copyright (c) 2019-2020, Alexander Breuer
 * Copyright (c) 2016-2018, Regents of the University of California
 * Copyright (c) 2016, Intel Corporation
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
 * 3. Neither the name of the copyright holder nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 * Optimized quadrature-free ADER-DG surface integration for single seismic
 *forward simulations.
 **/
#ifndef EDGE_SEISMIC_KERNELS_SURF_INT_SINGLE_BF16_HPP
#define EDGE_SEISMIC_KERNELS_SURF_INT_SINGLE_BF16_HPP

#include "SurfInt.hpp"
#include "data/MmXsmmSingle.hpp"
#include "dg/Basis.h"
#include <immintrin.h>


#define FP32_TO_BF16_SPLIT(input,out0,out1,out2) \
{ \
    out0 = _mm512_maskz_mov_epi16(0xAAAAAAAA,_mm512_castps_si512(input)); \
    __m512 _temp = _mm512_sub_ps(input,_mm512_castsi512_ps(out0)); \
    out1 = _mm512_maskz_mov_epi16(0xAAAAAAAA,_mm512_castps_si512(_temp)); \
    _temp = _mm512_sub_ps(_temp,_mm512_castsi512_ps(out1)); \
    out2 = _mm512_maskz_mov_epi16(0xAAAAAAAA,_mm512_castps_si512(_temp)); \
}

namespace edge
{
namespace seismic
{
namespace kernels
{
template <typename TL_T_REAL, unsigned short TL_N_RMS, t_entityType TL_T_EL,
          unsigned short TL_O_SP>
class SurfIntSingleBF16;
}
}
}

/**
 * Optimized quadrature-free ADER-DG surface integration for single seismic
 *forward simulations.
 *
 * @paramt TL_T_REAL floating point precision.
 * @paramt TL_N_RMS number of relaxation mechanisms.
 * @paramt TL_T_EL element type.
 * @paramt TL_O_SP spatial order.
 **/
template <typename TL_T_REAL, unsigned short TL_N_RMS, t_entityType TL_T_EL,
          unsigned short TL_O_SP>
class edge::seismic::kernels::SurfIntSingleBF16
    : public edge::seismic::kernels::SurfInt<TL_T_REAL, TL_N_RMS, TL_T_EL,
                                             TL_O_SP, 1>
{
private:
  //! number of dimensions
  static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

  //! number of faces
  static unsigned short const TL_N_FAS = C_ENT[TL_T_EL].N_FACES;

  //! number of DG face modes
  static unsigned short const TL_N_MDS_FA
      = CE_N_ELEMENT_MODES (C_ENT[TL_T_EL].TYPE_FACES, TL_O_SP);
  //! padded number of DG face modes
  static unsigned short const TL_N_MDS_FA_PAD = EVEN_PAD (TL_N_MDS_FA);

  //! number of DG element modes
  static unsigned short const TL_N_MDS_EL
      = CE_N_ELEMENT_MODES (TL_T_EL, TL_O_SP);
  //! padded number of DG element modes
  static unsigned short const TL_N_MDS_EL_PAD = EVEN_PAD (TL_N_MDS_EL);

  //! number of neigboring contribution flux matrices
  static unsigned short const TL_N_FMNS = CE_N_FLUXN_MATRICES (TL_T_EL);

  //! number of elastic quantities
  static unsigned short const TL_N_QTS_E = CE_N_QTS_E (TL_N_DIS);
  //! padded number of elastic quantities
  static unsigned short const TL_N_QTS_E_PAD = EVEN_PAD (TL_N_QTS_E);

  //! number of quantities per relaxation mechanism
  static unsigned short const TL_N_QTS_M = CE_N_QTS_M (TL_N_DIS);

  //! number of entries in the elastic flux solvers
  static unsigned short const TL_N_ENS_FS_E = CE_N_ENS_FS_E_DE (TL_N_DIS);

  //! number of entries in the anelastic flux solvers
  static unsigned short const TL_N_ENS_FS_A = CE_N_ENS_FS_A_DE (TL_N_DIS);

  //! pointers to the local flux matrices
  TL_T_REAL *m_fIntLN[TL_N_FAS + TL_N_FMNS] = {};
  TL_T_REAL l_fIntLN[TL_N_FAS + TL_N_FMNS][TL_N_MDS_EL_PAD * TL_N_MDS_FA]={};
  libxsmm_bfloat16 l_fIntLN_bf0[TL_N_FAS + TL_N_FMNS][TL_N_MDS_EL_PAD * TL_N_MDS_FA] = {};
  libxsmm_bfloat16 l_fIntLN_bf1[TL_N_FAS + TL_N_FMNS][TL_N_MDS_EL_PAD * TL_N_MDS_FA] = {};
  libxsmm_bfloat16 l_fIntLN_bf2[TL_N_FAS + TL_N_FMNS][TL_N_MDS_EL_PAD * TL_N_MDS_FA] = {};

  //! pointers to the transposed flux matrices
  TL_T_REAL *m_fIntT[TL_N_FAS] = {};

  TL_T_REAL l_fIntT[TL_N_FAS][TL_N_MDS_FA_PAD * TL_N_MDS_EL] = {};
  libxsmm_bfloat16 l_fIntT_bf0[TL_N_FAS][TL_N_MDS_FA_PAD * TL_N_MDS_EL] = {};
  libxsmm_bfloat16 l_fIntT_bf1[TL_N_FAS][TL_N_MDS_FA_PAD * TL_N_MDS_EL] = {};
  libxsmm_bfloat16 l_fIntT_bf2[TL_N_FAS][TL_N_MDS_FA_PAD * TL_N_MDS_EL] = {};

  //! matrix kernels
  edge::data::MmXsmmSingle<TL_T_REAL> m_fp;
  edge::data::MmXsmmSingle<libxsmm_bfloat16> m_bf;
  /**
   * Generates the matrix kernels for the flux matrices and flux solvers.
   **/
  void
  generateKernels ()
  {
    // add first flux matrix
    //  (0,0)
    m_fp.add (0,                            // group
              TL_N_MDS_FA,                  // m
              TL_N_QTS_E,                   // n
              TL_N_MDS_EL,                  // k
              TL_N_MDS_FA,                  // ldA
              TL_N_MDS_EL,                  // ldB
              TL_N_MDS_FA,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (0.0), // beta
              LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD);
    // (0,0)
    m_bf.add (0,                            // group
              TL_N_MDS_FA,                  // m
              TL_N_QTS_E,                   // n
              TL_N_MDS_EL_PAD,              // k
              TL_N_MDS_FA,                  // ldA
              TL_N_MDS_EL_PAD,              // ldB
              TL_N_MDS_FA,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (0.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);
    // (0,1)
    m_bf.add (0,                            // group
              TL_N_MDS_FA,                  // m
              TL_N_QTS_E,                   // n
              TL_N_MDS_EL_PAD,              // k
              TL_N_MDS_FA,                  // ldA
              TL_N_MDS_EL_PAD,              // ldB
              TL_N_MDS_FA,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (1.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);

    // add flux solver
    // (0,1)
    m_fp.add (0,                            // group
              TL_N_MDS_FA,                  // m
              TL_N_QTS_E,                   // n
              TL_N_QTS_E,                   // k
              TL_N_MDS_FA,                  // ldA
              TL_N_QTS_E,                   // ldB
              TL_N_MDS_FA,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (0.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);
    // (1,0)
    m_bf.add (1,                            // group
              TL_N_MDS_FA,                  // m
              TL_N_QTS_E,                   // n
              TL_N_QTS_E_PAD,               // k
              TL_N_MDS_FA,                  // ldA
              TL_N_QTS_E_PAD,               // ldB
              TL_N_MDS_FA,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (0.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);
    // (1,1)
    m_bf.add (1,                            // group
              TL_N_MDS_FA,                  // m
              TL_N_QTS_E,                   // n
              TL_N_QTS_E_PAD,               // k
              TL_N_MDS_FA,                  // ldA
              TL_N_QTS_E_PAD,               // ldB
              TL_N_MDS_FA,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (1.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);

    // add second flux matrix
    // (0,2)
    m_fp.add (0,                            // group
              TL_N_MDS_EL,                  // m
              TL_N_QTS_E,                   // n
              TL_N_MDS_FA,                  // k
              TL_N_MDS_EL,                  // ldA
              TL_N_MDS_FA,                  // ldB
              TL_N_MDS_EL,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (1.0), // beta
              LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD);
    // (2,0)
    m_bf.add (2,                            // group
              TL_N_MDS_EL,                  // m
              TL_N_QTS_E,                   // n
              TL_N_MDS_FA_PAD,              // k
              TL_N_MDS_EL,                  // ldA
              TL_N_MDS_FA_PAD,              // ldB
              TL_N_MDS_EL,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (1.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);

    // add anelastic flux solver
    m_fp.add (1,                            // group
              TL_N_MDS_FA,                  // m
              TL_N_QTS_M,                   // n
              TL_N_QTS_E,                   // k
              TL_N_MDS_FA,                  // ldA
              TL_N_QTS_E,                   // ldB
              TL_N_MDS_FA,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (0.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);

    // add second flux matrix for anelastic update
    m_fp.add (1,                            // group
              TL_N_MDS_EL,                  // m
              TL_N_QTS_M,                   // n
              TL_N_MDS_FA,                  // k
              TL_N_MDS_EL,                  // ldA
              TL_N_MDS_FA,                  // ldB
              TL_N_MDS_EL,                  // ldC
              static_cast<real_base> (1.0), // alpha
              static_cast<real_base> (1.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);
  }

public:
  /**
   * Constructor of the surface integrations for single forward simulations.
   *
   * @param io_dynMem dynamic memory allocations.
   **/
  SurfIntSingleBF16 (TL_T_REAL const *i_rfs, data::Dynamic &io_dynMem)
      : SurfInt<TL_T_REAL, TL_N_RMS, TL_T_EL, TL_O_SP, 1> (i_rfs, io_dynMem)
  {
    // store flux matrices dense
    this->storeFluxDense (io_dynMem, m_fIntLN, m_fIntT);

    for (unsigned short l_fas_flux = 0; l_fas_flux < TL_N_FAS + TL_N_FMNS;
         l_fas_flux++)
      {
        vnni_swap (m_fIntLN[l_fas_flux], &l_fIntLN[l_fas_flux][0], TL_N_MDS_EL,
                   TL_N_MDS_FA);
      }
    gen_bf_matrices<(TL_N_FAS + TL_N_FMNS) * TL_N_MDS_EL_PAD * TL_N_MDS_FA> (&l_fIntLN[0][0], &l_fIntLN_bf0[0][0], &l_fIntLN_bf1[0][0],
                     &l_fIntLN_bf2[0][0]);

    for (unsigned short l_fas = 0; l_fas < TL_N_FAS; l_fas++)
      {
        vnni_swap (m_fIntT[l_fas], &l_fIntT[l_fas][0], TL_N_MDS_FA,
                   TL_N_MDS_EL);
      }
    gen_bf_matrices<TL_N_FAS * TL_N_MDS_FA_PAD * TL_N_MDS_EL> (&l_fIntT[0][0], &l_fIntT_bf0[0][0], &l_fIntT_bf1[0][0],
                     &l_fIntT_bf2[0][0]);

    

    // generate kernels
    generateKernels ();
  }

  /**
   * Element local contribution for single forward simulations.
   *
   * @param i_fsE elastic flux solvers.
   * @param i_fsA anelastic flux solvers, use nullptr if TL_N_RMS==0.
   * @param i_tDofsE elastic time integerated DG-DOFs.
   * @param io_dofsE will be updated with local elastic contribution of the
   *element to the surface integral.
   * @param io_dofsA will be updated with local anelastic contribution of the
   *element to the surface integral, use nullptr for TL_N_RMS==0.
   * @param o_scratch will be used as scratch space for the computations.
   * @param i_dofsP DOFs for prefetching (not used).
   * @param i_tDofsP time integrated DOFs for prefetching (not used).
   **/
  void
  local (TL_T_REAL const i_fsE[TL_N_FAS][TL_N_ENS_FS_E],
         TL_T_REAL const (*i_fsA)[TL_N_ENS_FS_A],
         TL_T_REAL const i_tDofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
         TL_T_REAL io_dofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
         TL_T_REAL (*io_dofsA)[TL_N_QTS_M][TL_N_MDS_EL][1],
         TL_T_REAL o_scratch[2][TL_N_QTS_E][TL_N_MDS_FA][1],
         TL_T_REAL const i_dofsP[TL_N_QTS_E][TL_N_MDS_EL][1] = nullptr,
         TL_T_REAL const i_tDofsP[TL_N_QTS_E][TL_N_MDS_EL][1] = nullptr) const
  {
    // anelastic buffer
    TL_T_REAL l_upAn[TL_N_QTS_M][TL_N_MDS_EL][1];
    if (TL_N_RMS > 0)
      {
        for (unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++)
          {
#pragma omp simd
            for (unsigned short l_md = 0; l_md < TL_N_MDS_EL; l_md++)
              {
                l_upAn[l_qt][l_md][0] = 0;
              }
          }
      }
      
    TL_T_REAL l_scratch[TL_N_QTS_E * TL_N_MDS_FA_PAD] = {};
    libxsmm_bfloat16 l_scratch_bf0[TL_N_QTS_E * TL_N_MDS_FA_PAD] = {};
    libxsmm_bfloat16 l_scratch_bf1[TL_N_QTS_E * TL_N_MDS_FA_PAD] = {};
    libxsmm_bfloat16 l_scratch_bf2[TL_N_QTS_E * TL_N_MDS_FA_PAD] = {};

    TL_T_REAL l_tDofsE[TL_N_QTS_E * TL_N_MDS_EL_PAD] = {};
    libxsmm_bfloat16 l_tDofsE_bf0[TL_N_QTS_E * TL_N_MDS_EL_PAD] = {};
    libxsmm_bfloat16 l_tDofsE_bf1[TL_N_QTS_E * TL_N_MDS_EL_PAD] = {};
    libxsmm_bfloat16 l_tDofsE_bf2[TL_N_QTS_E * TL_N_MDS_EL_PAD] = {};

    dim_padding (&i_tDofsE[0][0][0], l_tDofsE, TL_N_MDS_EL, TL_N_QTS_E);
    gen_bf_matrices<TL_N_QTS_E * TL_N_MDS_EL_PAD>(l_tDofsE, l_tDofsE_bf0, l_tDofsE_bf1, l_tDofsE_bf2);
    //split_compress(l_tDofsE, l_tDofsE_bf0, l_tDofsE_bf1, l_tDofsE_bf2,
    //                 TL_N_QTS_E * TL_N_MDS_EL_PAD);

    // iterate over faces
    for (unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++)
      {

        // multiply with first face integration matrix
         //m_fp.m_kernels[0][0](m_fIntLN[l_fa], i_tDofsE[0][0],o_scratch[0][0][0], nullptr, i_dofsP[0][0], nullptr);

#if PP_APPROX_LEVEL == 0
        m_bf.m_kernels[0][0](l_fIntLN_bf0[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
#elif PP_APPROX_LEVEL == 1
        m_bf.m_kernels[0][0](l_fIntLN_bf0[l_fa], l_tDofsE_bf1, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf1[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
#elif PP_APPROX_LEVEL == 2
        m_bf.m_kernels[0][0](l_fIntLN_bf1[l_fa], l_tDofsE_bf1, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf2, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf2[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf1, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf1[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
#elif PP_APPROX_LEVEL == 3
        m_bf.m_kernels[0][0](l_fIntLN_bf2[l_fa], l_tDofsE_bf1, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf1[l_fa], l_tDofsE_bf2, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf1[l_fa], l_tDofsE_bf1, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf2, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf2[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf1, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf1[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
#elif PP_APPROX_LEVEL == 4
        m_bf.m_kernels[0][0](l_fIntLN_bf2[l_fa], l_tDofsE_bf2, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf2[l_fa], l_tDofsE_bf1, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf1[l_fa], l_tDofsE_bf2, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf1[l_fa], l_tDofsE_bf1, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf2, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf2[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf1, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf1[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
        m_bf.m_kernels[0][1](l_fIntLN_bf0[l_fa], l_tDofsE_bf0, o_scratch[0][0][0]);
#endif

        // multiply with flux solver
        m_fp.m_kernels[0][1](o_scratch[0][0][0], i_fsE[l_fa],
                             o_scratch[1][0][0]);

        dim_padding (&o_scratch[1][0][0][0], l_scratch, TL_N_MDS_FA,
                     TL_N_QTS_E);
        gen_bf_matrices<TL_N_QTS_E * TL_N_MDS_FA_PAD>(l_scratch, l_scratch_bf0, l_scratch_bf1,
                         l_scratch_bf2);
        //split_compress (l_scratch, l_scratch_bf0, l_scratch_bf1,
        //                 l_scratch_bf2, TL_N_QTS_E * TL_N_MDS_FA_PAD);

        // multiply with second face integration matrix
         //m_fp.m_kernels[0][2](m_fIntT[l_fa], o_scratch[1][0][0],
         //     io_dofsE[0][0],nullptr, i_tDofsP[0][0], nullptr);

#if PP_APPROX_LEVEL == 0
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf0, io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 1
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf1, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf1[l_fa], l_scratch_bf0, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf0, io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 2
        m_bf.m_kernels[2][0](l_fIntT_bf1[l_fa], l_scratch_bf1, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf2, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf2[l_fa], l_scratch_bf0, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf1, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf1[l_fa], l_scratch_bf0, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf0, io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 3
        m_bf.m_kernels[2][0](l_fIntT_bf2[l_fa], l_scratch_bf1, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf1[l_fa], l_scratch_bf2, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf1[l_fa], l_scratch_bf1, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf2, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf2[l_fa], l_scratch_bf0, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf1, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf1[l_fa], l_scratch_bf0, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf0, io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 4
        m_bf.m_kernels[2][0](l_fIntT_bf2[l_fa], l_scratch_bf2, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf2[l_fa], l_scratch_bf1, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf1[l_fa], l_scratch_bf2, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf1[l_fa], l_scratch_bf1, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf2, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf2[l_fa], l_scratch_bf0, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf1, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf1[l_fa], l_scratch_bf0, io_dofsE[0][0]);
        m_bf.m_kernels[2][0](l_fIntT_bf0[l_fa], l_scratch_bf0, io_dofsE[0][0]);
#endif

        if (TL_N_RMS > 0)
          {
            // multiply with anelastic flux solver
            m_fp.m_kernels[1][0](o_scratch[0][0][0], i_fsA[l_fa],
                                 o_scratch[1][0][0]);

            // multiply with secand face integration matrix
            m_fp.m_kernels[1][1](m_fIntT[l_fa], o_scratch[1][0][0],
                                 l_upAn[0][0]);
          }
      }

    // scatter to anelastic DOFs
    if (TL_N_RMS > 0)
      this->scatterUpdateA (l_upAn, io_dofsA);
  }

  /**
   * Applies the first first face-integration matrix to the elastic DOFs.
   *
   * @param i_fa local face.
   * @param i_vId id of the vertex, matching the element's vertex 0, from the
   *perspective of the adjacent element w.r.t. to the reference element.
   * @param i_fId id of the face from the perspective of the adjacent element
   *w.r.t. to the reference element.
   * @param i_tDofsE elastic time integrated DOFs.
   * @param o_tDofsFiE elastic time integrated DOFs after application of the
   *first face-int matrix.
   * @param i_pre DOFs or tDOFs for prefetching.
   **/
  void
  neighFluxInt (unsigned short i_fa, unsigned short i_vId,
                unsigned short i_fId,
                TL_T_REAL const i_tDofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
                TL_T_REAL o_tDofsFiE[TL_N_QTS_E][TL_N_MDS_FA][1],
                TL_T_REAL const i_pre[TL_N_QTS_E][TL_N_MDS_EL][1]
                = nullptr) const
  {
    // derive the id of the neighboring flux matrix
    unsigned short l_fMatId = std::numeric_limits<unsigned short>::max ();
    if (i_vId != std::numeric_limits<unsigned short>::max ())
      {
        l_fMatId = TL_N_FAS + this->fMatId (i_vId, i_fId);
      }
    else
      {
        l_fMatId = i_fa;
      }

    // multiply with first face integration matrix
    m_fp.m_kernels[0][0](m_fIntLN[l_fMatId], i_tDofsE[0][0], o_tDofsFiE[0][0],
                         nullptr, (TL_T_REAL const *)i_pre, nullptr);
  }

  /**
   * Neighboring contribution of a single adjacent element for single forward
   *simulations.
   *
   * @param i_fa local face.
   * @param i_vId id of the vertex, matching the element's vertex 0, from the
   *perspective of the adjacent element w.r.t. to the reference element.
   * @param i_fId id of the face from the perspective of the adjacent element
   *w.r.t. to the reference element.
   * @param i_fsE elastic flux solver.
   * @param i_fsA anelastic flux solver
   * @param i_tDofsE elastic time integrated DG-DOFs.
   * @param i_tDofsFiE elastic time integrated DG-DOFs multiplied with first
   *flux integration matrix.
   * @param io_dofsE will be updated with the elastic contribution of the
   *adjacent element to the surface integral.
   * @param io_dofsA will be updated with the unscaled (w.r.t. frequencies)
   *anelastic contribution of the adjacent element tot the surface integral,
   *use nullptr for TL_N_RMS==0.
   * @param o_scratch will be used as scratch space for the computations.
   * @param i_pre DOFs or tDOFs for prefetching.
   **/
  void
  neigh (unsigned short i_fa, unsigned short i_vId, unsigned short i_fId,
         TL_T_REAL const i_fsE[TL_N_ENS_FS_E],
         TL_T_REAL const i_fsA[TL_N_ENS_FS_A],
         TL_T_REAL const i_tDofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
         TL_T_REAL const i_tDofsFiE[TL_N_QTS_E][TL_N_MDS_FA][1],
         TL_T_REAL io_dofsE[TL_N_QTS_E][TL_N_MDS_EL][1],
         TL_T_REAL io_dofsA[TL_N_QTS_M][TL_N_MDS_EL][1],
         TL_T_REAL o_scratch[2][TL_N_QTS_E][TL_N_MDS_FA][1],
         TL_T_REAL const i_pre[TL_N_QTS_E][TL_N_MDS_EL][1] = nullptr) const
  {
    // apply first face integration matrix or set pointer to pre-computed data
    TL_T_REAL const *l_tDofsFiE = o_scratch[0][0][0];

    if (i_tDofsFiE == nullptr)
      {
        neighFluxInt (i_fa, i_vId, i_fId, i_tDofsE, o_scratch[0], i_pre);
      }
    else
      {
        l_tDofsFiE = i_tDofsFiE[0][0];
      }

    TL_T_REAL l_tDofsFiE_bf[TL_N_QTS_E_PAD * TL_N_MDS_FA] = {};
    libxsmm_bfloat16 l_tDofsFiE_bf0[TL_N_QTS_E_PAD * TL_N_MDS_FA] = {};
    libxsmm_bfloat16 l_tDofsFiE_bf1[TL_N_QTS_E_PAD * TL_N_MDS_FA] = {};
    libxsmm_bfloat16 l_tDofsFiE_bf2[TL_N_QTS_E_PAD * TL_N_MDS_FA] = {};

    TL_T_REAL l_fsE[TL_N_QTS_E * TL_N_QTS_E_PAD] = {};
    libxsmm_bfloat16 l_fsE_bf0[TL_N_QTS_E * TL_N_QTS_E_PAD] = {};
    libxsmm_bfloat16 l_fsE_bf1[TL_N_QTS_E * TL_N_QTS_E_PAD] = {};
    libxsmm_bfloat16 l_fsE_bf2[TL_N_QTS_E * TL_N_QTS_E_PAD] = {};

    vnni_swap (l_tDofsFiE, l_tDofsFiE_bf, TL_N_QTS_E, TL_N_MDS_FA);

    gen_bf_matrices<TL_N_QTS_E_PAD * TL_N_MDS_FA>(l_tDofsFiE_bf, l_tDofsFiE_bf0, l_tDofsFiE_bf1, l_tDofsFiE_bf2);
    //split_compress(l_tDofsFiE_bf, l_tDofsFiE_bf0, l_tDofsFiE_bf1,
    //                 l_tDofsFiE_bf2, TL_N_QTS_E_PAD * TL_N_MDS_FA);

    dim_padding (&i_fsE[0], l_fsE, TL_N_QTS_E, TL_N_QTS_E);
    gen_bf_matrices<TL_N_QTS_E_PAD * TL_N_QTS_E>(l_fsE, l_fsE_bf0, l_fsE_bf1, l_fsE_bf2);
    //split_compress(l_fsE, l_fsE_bf0, l_fsE_bf1, l_fsE_bf2,
    //                TL_N_QTS_E_PAD * TL_N_QTS_E);

// multiply with flux solver
// m_fp.m_kernels[0][1](l_tDofsFiE, i_fsE, o_scratch[1][0][0]);
#if PP_APPROX_LEVEL == 0
    m_bf.m_kernels[1][0](l_tDofsFiE_bf0, l_fsE_bf0, o_scratch[1][0][0]);
#elif PP_APPROX_LEVEL == 1
    m_bf.m_kernels[1][0](l_tDofsFiE_bf0, l_fsE_bf1, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf1, l_fsE_bf0, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf0, o_scratch[1][0][0]);
#elif PP_APPROX_LEVEL == 2
    m_bf.m_kernels[1][0](l_tDofsFiE_bf1, l_fsE_bf1, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf2, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf2, l_fsE_bf0, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf1, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf1, l_fsE_bf0, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf0, o_scratch[1][0][0]);
#elif PP_APPROX_LEVEL == 3
    m_bf.m_kernels[1][0](l_tDofsFiE_bf2, l_fsE_bf1, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf1, l_fsE_bf2, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf1, l_fsE_bf1, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf2, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf2, l_fsE_bf0, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf1, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf1, l_fsE_bf0, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf0, o_scratch[1][0][0]);
#elif PP_APPROX_LEVEL == 4
    m_bf.m_kernels[1][0](l_tDofsFiE_bf2, l_fsE_bf2, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf2, l_fsE_bf1, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf1, l_fsE_bf2, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf1, l_fsE_bf1, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf2, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf2, l_fsE_bf0, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf1, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf1, l_fsE_bf0, o_scratch[1][0][0]);
    m_bf.m_kernels[1][1](l_tDofsFiE_bf0, l_fsE_bf0, o_scratch[1][0][0]);
#endif
    TL_T_REAL l_scratch[TL_N_QTS_E * TL_N_MDS_FA_PAD] = {};
    libxsmm_bfloat16 l_scratch_bf0[TL_N_QTS_E * TL_N_MDS_FA_PAD] = {};
    libxsmm_bfloat16 l_scratch_bf1[TL_N_QTS_E * TL_N_MDS_FA_PAD] = {};
    libxsmm_bfloat16 l_scratch_bf2[TL_N_QTS_E * TL_N_MDS_FA_PAD] = {};

    dim_padding (&o_scratch[1][0][0][0], l_scratch, TL_N_MDS_FA, TL_N_QTS_E);
    gen_bf_matrices <TL_N_QTS_E * TL_N_MDS_FA_PAD>(l_scratch, l_scratch_bf0, l_scratch_bf1, l_scratch_bf2);
    //split_compress(l_scratch, l_scratch_bf0, l_scratch_bf1, l_scratch_bf2,
    //                TL_N_QTS_E * TL_N_MDS_FA_PAD);

    // multiply with second face integration matrix
     // m_fp.m_kernels[0][2](m_fIntT[i_fa], o_scratch[1][0][0], io_dofsE[0][0],
     //                    nullptr, (TL_T_REAL const *)i_pre, nullptr);

#if PP_APPROX_LEVEL == 0
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf0, io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 1
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf1, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf1[i_fa], l_scratch_bf0, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf0, io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 2
    m_bf.m_kernels[2][0](l_fIntT_bf1[i_fa], l_scratch_bf1, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf2, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf2[i_fa], l_scratch_bf0, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf1, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf1[i_fa], l_scratch_bf0, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf0, io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 3
    m_bf.m_kernels[2][0](l_fIntT_bf2[i_fa], l_scratch_bf1, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf1[i_fa], l_scratch_bf2, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf1[i_fa], l_scratch_bf1, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf2, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf2[i_fa], l_scratch_bf0, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf1, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf1[i_fa], l_scratch_bf0, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf0, io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 4
    m_bf.m_kernels[2][0](l_fIntT_bf2[i_fa], l_scratch_bf2, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf2[i_fa], l_scratch_bf1, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf1[i_fa], l_scratch_bf2, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf1[i_fa], l_scratch_bf1, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf2, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf2[i_fa], l_scratch_bf0, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf1, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf1[i_fa], l_scratch_bf0, io_dofsE[0][0]);
    m_bf.m_kernels[2][0](l_fIntT_bf0[i_fa], l_scratch_bf0, io_dofsE[0][0]);
#endif

    if (TL_N_RMS > 0)
      {
        // multiply with anelastic flux solver
        m_fp.m_kernels[1][0](l_tDofsFiE, i_fsA, o_scratch[1][0][0]);

        // multiply with second face integration matrix
        m_fp.m_kernels[1][1](m_fIntT[i_fa], o_scratch[1][0][0],
                             io_dofsA[0][0]);
      }
  }

  // add 0 in B. dst must be filed with zeros and of size (K+1)(N)
  template <typename T>
  void
  dim_padding (const T *src, T *dest, const unsigned short &K,
               const unsigned short &N) const
  {
    if (K % 2 == 1)
      {
        for (size_t l_n = 0; l_n < N; l_n++)
          {
#pragma omp simd
            for (size_t l_k = 0; l_k < K; l_k++)
              {
                dest[l_k + l_n * (K + 1)] = src[l_k + l_n * K];
              }
          }
      }
    else
      {
        for (size_t l_n = 0; l_n < N; l_n++)
          {
#pragma omp simd
            for (size_t l_k = 0; l_k < K; l_k++)
              {
                dest[l_k + l_n * K] = src[l_k + l_n * K];
              }
          }
      }
  }

  template <typename T>
  void
  vnni_swap (T const *i_src, T *io_dest, size_t K, size_t M) const
  {
    for (size_t l_k = 0; l_k < K; l_k += 2)
      {
        for (size_t l_m = 0; l_m < M; l_m++)
          {
            io_dest[2 * l_m + l_k * M] = i_src[l_m + l_k * M];
            io_dest[2 * l_m + l_k * M + 1] = i_src[l_m + (l_k + 1) * M];
          }
      }

    if (K % 2 == 1)
      {
        for (size_t l_m = 0; l_m < M; l_m++)
          {
            io_dest[2 * l_m + (K - 1) * M] = i_src[l_m + (K - 1) * M];
            io_dest[2 * l_m + (K - 1) * M + 1] = 0;
          }
      }
  }
template <unsigned int size>
  void
  gen_bf_matrices (const float *src, libxsmm_bfloat16 *bf_0,
                   libxsmm_bfloat16 *bf_1, libxsmm_bfloat16 *bf_2) const
  {

    libxsmm_rne_convert_fp32_bf16 (src, bf_0, size);

  
    float intermediate[size] = {};
    float copy[size] = {};  

    //libxsmm_truncate_convert_f32_bf16(src, bf_0, size);
    libxsmm_convert_bf16_f32 (bf_0, copy, size);
    
    #pragma omp simd
        for (size_t i = 0; i < size; i++)
          {
            intermediate[i] = src[i] - copy[i];
          }
    
    libxsmm_rne_convert_fp32_bf16 (intermediate, bf_1, size);
    //libxsmm_truncate_convert_f32_bf16(intermediate, bf_1, size);
  
    libxsmm_convert_bf16_f32 (bf_1, copy, size);
    
    #pragma omp simd
        for (size_t i = 0; i < size; i++)
          {
            intermediate[i] = intermediate[i] - copy[i];
          }
          
    libxsmm_rne_convert_fp32_bf16 (intermediate, bf_2, size);
    //libxsmm_truncate_convert_f32_bf16(intermediate, bf_2, size);
 
  }

  void split_compress(const float *input,libxsmm_bfloat16 *output0, libxsmm_bfloat16 *output1, libxsmm_bfloat16 *output2, 
                      const int size) const
  {
    static const unsigned short perm_idx_buffer[] = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,
                                        0x20|1,0x20|3,0x20|5,0x20|7,0x20|9,0x20|11,0x20|13,0x20|15,
                                        0x20|17,0x20|19,0x20|21,0x20|23,0x20|25,0x20|27,0x20|29,0x20|31};

    static const __m512i PERM_IDX = _mm512_loadu_si512(perm_idx_buffer);

    for(int i=0;i<size;i+=32)
    {
        __m512 fp_inputL = _mm512_loadu_ps(input + i);
        __m512 fp_inputH = _mm512_loadu_ps(input + 16 + i);

        __m512i inL[3];
        __m512i inH[3];

        FP32_TO_BF16_SPLIT(fp_inputL,inL[0],inL[1],inL[2]);
        FP32_TO_BF16_SPLIT(fp_inputH,inH[0],inH[1],inH[2]);

        __m512i in0 = _mm512_permutex2var_epi16 (inL[0],PERM_IDX, inH[0]);
        __m512i in1 = _mm512_permutex2var_epi16 (inL[1],PERM_IDX, inH[1]);
        __m512i in2 = _mm512_permutex2var_epi16 (inL[2],PERM_IDX, inH[2]);

        _mm512_storeu_si512(output0,in0);
        _mm512_storeu_si512(output1,in1);
        _mm512_storeu_si512(output2,in2);
        output0+=32;
        output1+=32;
        output2+=32;
    }
    
  }
};

#endif