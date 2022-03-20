/**
 * @file This file is part of EDGE.
 *
 * @author Alexander Breuer (anbreuer AT ucsd.edu)
 *         Alexander Heinecke (alexander.heinecke AT intel.com)
 *         Julius Isken
 *
 * @section LICENSE
 * Copyright (c) 2019, Alexander Breuer
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
 * Optimized quadrature-free ADER-DG volume integration for single seismic wave
 *propagation.
 **/
#ifndef EDGE_SEISMIC_KERNELS_VOL_INT_SINGLE_BF16_HPP
#define EDGE_SEISMIC_KERNELS_VOL_INT_SINGLE_BF16_HPP

#include <immintrin.h>

#include <chrono>
#include <iostream>

#include "VolInt.hpp"
#include "data/MmXsmmSingle.hpp"
#include "dg/Basis.h"

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
class VolIntSingleBF16;
}
} // namespace seismic
} // namespace edge

/**
 * Optimized quadrature-free ADER-DG volume integration for single seismic
 *forward simulations.
 *
 * @paramt TL_T_REAL floating point precision.
 * @paramt TL_N_RMS number of relaxation mechanisms.
 * @paramt TL_T_EL element type.
 * @paramt TL_O_SP spatial order.
 **/
template <typename TL_T_REAL, unsigned short TL_N_RMS, t_entityType TL_T_EL,
          unsigned short TL_O_SP>
class edge::seismic::kernels::VolIntSingleBF16
    : edge::seismic::kernels::VolInt<TL_T_REAL, TL_N_RMS, TL_T_EL, TL_O_SP, 1>
{
private:
  //! dimension of the element
  static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

  //! number of element modes
  static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES (TL_T_EL, TL_O_SP);
  //! even padded number of element modes
  static unsigned short const TL_N_MDS_PAD = EVEN_PAD (TL_N_MDS);

  //! number of active element modes
  static unsigned short const TL_N_ACT_MDS
      = CE_N_ELEMENT_MODES_CK (TL_T_EL, TL_O_SP, 1);
  //! even padded number of active element modes
  static unsigned short const TL_N_ACT_MDS_PAD = EVEN_PAD (TL_N_ACT_MDS);

  //! number of elastic quantities
  static unsigned short const TL_N_QTS_E = CE_N_QTS_E (TL_N_DIS);
  //! even padded number of elastic quantities
  static unsigned short const TL_N_QTS_E_PAD = EVEN_PAD (TL_N_QTS_E);

  //! number of quantities per relaxation mechanism
  static unsigned short const TL_N_QTS_M = CE_N_QTS_M (TL_N_DIS);

  //! number of non-zeros in the elastic star matrices
  static unsigned short const TL_N_ENS_STAR_E = CE_N_ENS_STAR_E_DE (TL_N_DIS);

  //! number of non-zeros in the anelastic star matrices
  static unsigned short const TL_N_ENS_STAR_A = CE_N_ENS_STAR_A_DE (TL_N_DIS);

  //! number of non-zeros in the anelastic source matrices
  static unsigned short const TL_N_ENS_SRC_A = CE_N_ENS_SRC_A_DE (TL_N_DIS);

  //! matrix kernels
  edge::data::MmXsmmSingle<libxsmm_bfloat16> m_bf;
  edge::data::MmXsmmSingle<TL_T_REAL> m_fp;

  //! pointers to the stiffness matrices [TL_N_DIS][TL_N_MDS][TL_NMDS]
  TL_T_REAL *m_stiff[TL_N_DIS] = {};

  TL_T_REAL l_stiff[TL_N_DIS][TL_N_MDS_PAD * TL_N_MDS] = {};
  libxsmm_bfloat16 l_stiff_bf0[TL_N_DIS][TL_N_MDS_PAD * TL_N_MDS] = {};
  libxsmm_bfloat16 l_stiff_bf1[TL_N_DIS][TL_N_MDS_PAD * TL_N_MDS] = {};
  libxsmm_bfloat16 l_stiff_bf2[TL_N_DIS][TL_N_MDS_PAD * TL_N_MDS] = {};
  /**
   * Generates the matrix kernels for the stiffness matrices and star
   *matrices.
   **/
  void
  generateKernels ()
  {
    m_bf.add (0,                            // group
              TL_N_MDS,                     // m
              TL_N_QTS_E,                   // n
              TL_N_ACT_MDS_PAD,             // k
              TL_N_MDS,                     // ldA
              TL_N_MDS_PAD,                 // ldB
              TL_N_MDS,                     // ldC
              static_cast<TL_T_REAL> (1.0), // alpha
              static_cast<TL_T_REAL> (0.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);
    m_bf.add (0,                            // group
              TL_N_MDS,                     // m
              TL_N_QTS_E,                   // n
              TL_N_ACT_MDS_PAD,             // k
              TL_N_MDS,                     // ldA
              TL_N_MDS_PAD,                 // ldB
              TL_N_MDS,                     // ldC
              static_cast<TL_T_REAL> (1.0), // alpha
              static_cast<TL_T_REAL> (1.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);

    m_fp.add (0,                                           // group
              TL_N_MDS,                                    // m
              TL_N_QTS_E,                                  // n
              CE_N_ELEMENT_MODES_CK (TL_T_EL, TL_O_SP, 1), // k
              TL_N_MDS,                                    // ldA
              TL_N_MDS,                                    // ldB
              TL_N_MDS,                                    // ldC
              static_cast<TL_T_REAL> (1.0),                // alpha
              static_cast<TL_T_REAL> (0.0),                // beta
              LIBXSMM_GEMM_PREFETCH_NONE);

    m_fp.add (0,                            // group
              TL_N_MDS,                     // m
              TL_N_QTS_E,                   // n
              TL_N_QTS_E,                   // k
              TL_N_MDS,                     // ldA
              TL_N_QTS_E,                   // ldB
              TL_N_MDS,                     // ldC
              static_cast<TL_T_REAL> (1.0), // alpha
              static_cast<TL_T_REAL> (1.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);

    m_bf.add (0,                            // group
              TL_N_MDS,                     // m
              TL_N_QTS_E,                   // n
              TL_N_QTS_E_PAD,               // k
              TL_N_MDS,                     // ldA
              TL_N_QTS_E_PAD,               // ldB
              TL_N_MDS,                     // ldC
              static_cast<TL_T_REAL> (1.0), // alpha
              static_cast<TL_T_REAL> (1.0), // beta
              LIBXSMM_GEMM_PREFETCH_NONE);

    if (TL_N_RMS > 0)
      {
        // anelastic star matrix
        m_fp.add (1,                            // group
                  TL_N_MDS,                     // m
                  TL_N_QTS_M,                   // n
                  TL_N_DIS,                     // k
                  TL_N_MDS,                     // ldA
                  TL_N_DIS,                     // ldB
                  TL_N_MDS,                     // ldC
                  static_cast<TL_T_REAL> (1.0), // alpha
                  static_cast<TL_T_REAL> (1.0), // beta
                  LIBXSMM_GEMM_PREFETCH_NONE);

        // anelastic source matrix
        m_fp.add (1,                            // group
                  TL_N_MDS,                     // m
                  TL_N_QTS_M,                   // n
                  TL_N_QTS_M,                   // k
                  TL_N_MDS,                     // ldA
                  TL_N_QTS_M,                   // ldB
                  TL_N_MDS,                     // ldC
                  static_cast<TL_T_REAL> (1.0), // alpha
                  static_cast<TL_T_REAL> (1.0), // beta
                  LIBXSMM_GEMM_PREFETCH_NONE);
      }
  }

public:
  /**
   * Constructor of the optimized volume integration for single forward
   *simulations.
   *
   * @param i_rfs relaxation frequencies, use nullptr if TL_N_RMS==0.
   * @param io_dynMem dynamic memory allocations.
   **/
  VolIntSingleBF16 (TL_T_REAL const *i_rfs, data::Dynamic &io_dynMem)
      : VolInt<TL_T_REAL, TL_N_RMS, TL_T_EL, TL_O_SP, 1> (i_rfs, io_dynMem)
  {
    // store stiffness matrices dense
    this->storeStiffDense (io_dynMem, m_stiff);

    for (unsigned short l_di = 0; l_di < TL_N_DIS; l_di++)
      {
        vnni_swap (m_stiff[l_di], &l_stiff[l_di][0], TL_N_MDS, TL_N_MDS);

       
      }
       gen_bf_matrices<TL_N_MDS_PAD * TL_N_MDS * TL_N_DIS > (
            &l_stiff[0][0], &l_stiff_bf0[0][0], &l_stiff_bf1[0][0],
            &l_stiff_bf2[0][0]);

    // generate matrix kernels
    generateKernels ();
  }

  /**
   * Optimized volume contribution for single forward simulations.
   *
   * @param i_starE elastic star matrices.
   * @param i_starA anelastic star matrices, use nullptr if TL_N_RMS==0.
   * @param i_tDofsE time integrated elastic DOFs.
   * @param i_tDofsA time integrated anselastic DOFs.
   * @param io_dofsE will be updated with local elastic contribution of the
   *element to the volume integral.
   * @param io_dofsA will be updated with local anelastic contribution of the
   *element to the volume integral, use nullptr if TL_N_RMS==0.
   * @param o_scratch will be used as scratch space for the computations.
   **/
  void
  apply (TL_T_REAL const i_starE[TL_N_DIS][TL_N_ENS_STAR_E],
         TL_T_REAL const (*i_starA)[TL_N_ENS_STAR_A],
         TL_T_REAL const (*i_srcA)[TL_N_ENS_SRC_A],
         TL_T_REAL const i_tDofsE[TL_N_QTS_E][TL_N_MDS][1],
         TL_T_REAL const (*i_tDofsA)[TL_N_QTS_M][TL_N_MDS][1],
         TL_T_REAL io_dofsE[TL_N_QTS_E][TL_N_MDS][1],
         TL_T_REAL (*io_dofsA)[TL_N_QTS_M][TL_N_MDS][1],
         TL_T_REAL o_scratch[TL_N_QTS_E][TL_N_MDS][1])
  {
    // relaxation frequencies
    TL_T_REAL const *l_rfs = this->m_rfs;

    TL_T_REAL l_tDofsE[TL_N_MDS_PAD * TL_N_QTS_E] = {};
    libxsmm_bfloat16 l_tDofsE_bf0[TL_N_MDS_PAD * TL_N_QTS_E] = {};
    libxsmm_bfloat16 l_tDofsE_bf1[TL_N_MDS_PAD * TL_N_QTS_E] = {};
    libxsmm_bfloat16 l_tDofsE_bf2[TL_N_MDS_PAD * TL_N_QTS_E] = {};

    TL_T_REAL l_scratch_bf[TL_N_MDS * TL_N_QTS_E_PAD] = {};
    libxsmm_bfloat16 l_scratch_bf0[TL_N_MDS * TL_N_QTS_E_PAD] = {};
    libxsmm_bfloat16 l_scratch_bf1[TL_N_MDS * TL_N_QTS_E_PAD] = {};
    libxsmm_bfloat16 l_scratch_bf2[TL_N_MDS * TL_N_QTS_E_PAD] = {};

    TL_T_REAL l_starE[TL_N_DIS][TL_N_QTS_E * TL_N_QTS_E_PAD] = {};      
    libxsmm_bfloat16 l_starE_bf0[TL_N_DIS][TL_N_QTS_E * TL_N_QTS_E_PAD] = {};
    libxsmm_bfloat16 l_starE_bf1[TL_N_DIS][TL_N_QTS_E * TL_N_QTS_E_PAD] = {};    
    libxsmm_bfloat16 l_starE_bf2[TL_N_DIS][TL_N_QTS_E * TL_N_QTS_E_PAD] = {};
    

    
    
    // buffer for anelastic part
    TL_T_REAL l_scratch[TL_N_QTS_M][TL_N_MDS][1];

    for (unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++)
      {
        for (unsigned short l_md = 0; l_md < TL_N_MDS; l_md++)
          {
            l_scratch[l_qt][l_md][0] = 0;
          }
      }

    dim_padding (&i_tDofsE[0][0][0], l_tDofsE, TL_N_MDS, TL_N_QTS_E);
    gen_bf_matrices<TL_N_MDS_PAD * TL_N_QTS_E> (l_tDofsE, l_tDofsE_bf0,
                                                l_tDofsE_bf1, l_tDofsE_bf2);

    //split_compress(l_tDofsE, l_tDofsE_bf0, l_tDofsE_bf1, l_tDofsE_bf2,
                     TL_N_MDS_PAD * TL_N_QTS_E);
    // TODO For loop necessary?

    for (unsigned short l_di = 0; l_di < TL_N_DIS; l_di++)
      {
         dim_padding (&i_starE[l_di][0], l_starE[l_di], TL_N_QTS_E,
        TL_N_QTS_E);
        //split_compress( l_starE[l_di], l_starE_bf0[l_di], l_starE_bf1[l_di],
        //    l_starE_bf2[l_di], TL_N_QTS_E_PAD * TL_N_QTS_E);
        
      }
      gen_bf_matrices<TL_N_QTS_E_PAD * TL_N_QTS_E * TL_N_DIS> (
           l_starE[0], l_starE_bf0[0], l_starE_bf1[0],
            l_starE_bf2[0]);
      

    // iterate over dimensions
    for (unsigned short l_di = 0; l_di < TL_N_DIS; l_di++)
      {
        // stiffness and inverse mass matrix
        // m_fp.m_kernels[0][0](m_stiff[l_di], i_tDofsE[0][0],
        // o_scratch[0][0]);

#if PP_APPROX_LEVEL == 0
        m_bf.m_kernels[0][0](l_stiff_bf0[l_di], l_tDofsE_bf0, o_scratch[0][0]);
#elif PP_APPROX_LEVEL == 1
        m_bf.m_kernels[0][0](l_stiff_bf0[l_di], l_tDofsE_bf1, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf1[l_di], l_tDofsE_bf0, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf0, o_scratch[0][0]);
#elif PP_APPROX_LEVEL == 2
        m_bf.m_kernels[0][0](l_stiff_bf1[l_di], l_tDofsE_bf1, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf2, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf2[l_di], l_tDofsE_bf0, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf1, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf1[l_di], l_tDofsE_bf0, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf0, o_scratch[0][0]);
#elif PP_APPROX_LEVEL == 3
        m_bf.m_kernels[0][0](l_stiff_bf2[l_di], l_tDofsE_bf1, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf1[l_di], l_tDofsE_bf2, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf1[l_di], l_tDofsE_bf1, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf2, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf2[l_di], l_tDofsE_bf0, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf1, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf1[l_di], l_tDofsE_bf0, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf0, o_scratch[0][0]);
#elif PP_APPROX_LEVEL == 4
        m_bf.m_kernels[0][0](l_stiff_bf2[l_di], l_tDofsE_bf2, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf2[l_di], l_tDofsE_bf1, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf1[l_di], l_tDofsE_bf2, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf1[l_di], l_tDofsE_bf1, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf2, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf2[l_di], l_tDofsE_bf0, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf1, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf1[l_di], l_tDofsE_bf0, o_scratch[0][0]);
        m_bf.m_kernels[0][1](l_stiff_bf0[l_di], l_tDofsE_bf0, o_scratch[0][0]);
#endif

      vnni_swap (&o_scratch[0][0][0], l_scratch_bf, TL_N_QTS_E, TL_N_MDS);
      gen_bf_matrices<TL_N_MDS * TL_N_QTS_E_PAD> (
            l_scratch_bf, l_scratch_bf0, l_scratch_bf1, l_scratch_bf2);
      //split_compress(l_scratch_bf, l_scratch_bf0, l_scratch_bf1, l_scratch_bf2 , TL_N_MDS * TL_N_QTS_E_PAD);

        //m_fp.m_kernels[0][1](o_scratch[0][0], i_starE[l_di],
        // io_dofsE[0][0]);

        // m_fp.m_kernels[0][1](o_scratch[0][0], i_starE[l_di],
        // io_dofsE[0][0]);

#if PP_APPROX_LEVEL == 0
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf0[l_di], io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 1
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf1[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf1, l_starE_bf0[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf0[l_di], io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 2 
        m_bf.m_kernels[0][2](l_scratch_bf1, l_starE_bf1[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf2[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf2, l_starE_bf0[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf1[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf1, l_starE_bf0[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf0[l_di], io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 3 
        m_bf.m_kernels[0][2](l_scratch_bf2, l_starE_bf1[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf1, l_starE_bf2[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf1, l_starE_bf1[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf2[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf2, l_starE_bf0[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf1[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf1, l_starE_bf0[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf0[l_di], io_dofsE[0][0]);
#elif PP_APPROX_LEVEL == 4 
        m_bf.m_kernels[0][2](l_scratch_bf2, l_starE_bf2[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf2, l_starE_bf1[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf1, l_starE_bf2[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf1, l_starE_bf1[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf2[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf2, l_starE_bf0[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf1[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf1, l_starE_bf0[l_di], io_dofsE[0][0]);
        m_bf.m_kernels[0][2](l_scratch_bf0, l_starE_bf0[l_di], io_dofsE[0][0]);
#endif

            if (TL_N_RMS > 0)
        {
          // anelastic star matrix
          m_fp.m_kernels[1][0](o_scratch[TL_N_QTS_M][0], i_starA[l_di],
                               l_scratch[0][0]);
        }
        
      }

    for (unsigned short l_rm = 0; l_rm < TL_N_RMS; l_rm++)
      {
        // add contribution of source matrix
        m_fp.m_kernels[1][1](i_tDofsA[l_rm][0][0], i_srcA[l_rm],
                             io_dofsE[0][0]);

        // multiply with relaxation frequency and add
        for (unsigned short l_qt = 0; l_qt < TL_N_QTS_M; l_qt++)
          {
#pragma omp simd
            for (unsigned short l_md = 0; l_md < TL_N_MDS; l_md++)
              {
                io_dofsA[l_rm][l_qt][l_md][0]
                    += l_rfs[l_rm]
                       * (l_scratch[l_qt][l_md][0]
                          - i_tDofsA[l_rm][l_qt][l_md][0]);
              }
          }
      }
  }

  // add 0 in B. dst must be filed with zeros and of size (K+1)(N)

  void
  dim_padding (const float *src, float *dest, const unsigned short &K,
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
#pragma omp simd
        for (size_t l_m = 0; l_m < M; l_m++)
          {
            io_dest[2 * l_m + l_k * M] = i_src[l_m + l_k * M];
            io_dest[2 * l_m + l_k * M + 1] = i_src[l_m + (l_k + 1) * M];
          }
      }

    if (K % 2 == 1)
      {
#pragma omp simd
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
  void split_compress(const float *input,libxsmm_bfloat16 *output0, libxsmm_bfloat16 *output1, libxsmm_bfloat16 *output2, const int size)
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
  