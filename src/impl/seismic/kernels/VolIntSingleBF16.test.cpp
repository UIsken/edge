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
 * Tests the optimized volume integration for single forward simulations.
 **/
#include <catch.hpp>
#define private public
#include <chrono>

#include "VolIntSingle.hpp"
#include "VolIntSingleBF16.hpp"
#undef private

TEST_CASE ("Bfloat approximation for Volume Integration",
           "[elastic][VolIntSingle][bfloat][conversion]")
{
#include "VolInt.test.inc"

  // volume kernel
  edge::data::Dynamic l_dynMem;
  edge::seismic::kernels::VolIntSingleBF16<float, 0, TET4, 4> l_vol (nullptr,
                                                                     l_dynMem);

  float l_scratch[9][20][1];

  float bf_dofsE[9][20];
  float bf0_dofsE[2][9][20];
  float bf1_dofsE[2][9][20];
  float bf2_dofsE[2][9][20];
  libxsmm_bfloat16 l_tDofsE_bf0[2][9][20];
  libxsmm_bfloat16 l_tDofsE_bf1[2][9][20];
  libxsmm_bfloat16 l_tDofsE_bf2[2][9][20];

  l_vol.gen_bf_matrices<20 * 9> (&l_tDofsE[0][0], &l_tDofsE_bf0[0][0][0],
                                 &l_tDofsE_bf2[0][0][0], &l_tDofsE_bf2[0][0][0]);
  l_vol.split_compress(&l_tDofsE[0][0], &l_tDofsE_bf0[1][0][0],
                                 &l_tDofsE_bf2[1][0][0], &l_tDofsE_bf2[1][0][0], 20*9);
  for (size_t i = 0; i < 2; i++)
  {  
  libxsmm_convert_bf16_f32 (&l_tDofsE_bf0[i][0][0], &bf0_dofsE[i][0][0], 20 * 9);
  libxsmm_convert_bf16_f32 (&l_tDofsE_bf1[i][0][0], &bf1_dofsE[i][0][0], 20 * 9);
  libxsmm_convert_bf16_f32 (&l_tDofsE_bf1[i][0][0], &bf2_dofsE[i][0][0], 20 * 9);
  }

  for (size_t j = 0; j < 9; j++)
    {
      for (size_t i = 0; i < 20; i++)
        {
          //REQUIRE(bf0_dofsE[0][j][i]==bf0_dofsE[1][j][i]);
          //REQUIRE(bf1_dofsE[0][j][i]==bf1_dofsE[1][j][i]);
          //REQUIRE(bf2_dofsE[0][j][i]==bf2_dofsE[1][j][i]);
          //std::cout << "Split compress" << l_tDofsE_bf0[1][j][i] << std::endl;
          //std::cout << "naive method " << l_tDofsE_bf0[0][j][i] << std::endl;
          //REQUIRE(l_tDofsE_bf0[0][j][i]==l_tDofsE_bf0[1][j][i]);
          //REQUIRE(l_tDofsE_bf1[0][j][i]==l_tDofsE_bf1[1][j][i]);
          //REQUIRE(l_tDofsE_bf2[0][j][i]==l_tDofsE_bf2[1][j][i]);
        }
    }

  float bf_starE[3][9][9];
  float bf0_starE[3][9][9];
  float bf1_starE[3][9][9];
  float bf2_starE[3][9][9];
  libxsmm_bfloat16 l_starE_bf0[3][9][9];
  libxsmm_bfloat16 l_starE_bf1[3][9][9];
  libxsmm_bfloat16 l_starE_bf2[3][9][9];

  for (size_t i = 0; i < 3; i++)
    {
      l_vol.gen_bf_matrices<9 * 9> (&l_starE[i][0][0], &l_starE_bf0[i][0][0],
                                    &l_starE_bf1[i][0][0],
                                    &l_starE_bf2[i][0][0]);

      libxsmm_convert_bf16_f32 (&l_starE_bf0[i][0][0], &bf0_starE[i][0][0],
                                9 * 9);
      libxsmm_convert_bf16_f32 (&l_starE_bf1[i][0][0], &bf1_starE[i][0][0],
                                9 * 9);
      libxsmm_convert_bf16_f32 (&l_starE_bf1[i][0][0], &bf2_starE[i][0][0],
                                9 * 9);
    }
  for (size_t l = 0; l < 3; l++)
    {
      for (size_t i = 0; i < 9; i++)
        {
          for (size_t j = 0; j < 9; j++)
            {
              bf_starE[l][i][j] = bf0_starE[l][i][j] + bf1_starE[l][i][j]
                                  + bf2_starE[l][i][j];
            }
        }
    }

  for (size_t l = 0; l < 3; l++)
    {
      for (size_t i = 0; i < 9; i++)
        {
          for (size_t j = 0; j < 9; j++)
            {
              REQUIRE (bf_starE[l][i][j]
                       == Approx (l_starE[l][i][j]).epsilon (0.1));
            }
        }
    }
}

TEST_CASE (
    "Optimized elastic volume integration for single forward simulations with "
    "bfloat approximation.",
    "[elastic][VolIntSingle][order4][bfloat]")
{
  // set up matrix structures
#include "VolInt.test.inc"

  // volume kernel
  edge::data::Dynamic bf_dynMem;
  edge::data::Dynamic fp_dynMem;
  edge::seismic::kernels::VolIntSingleBF16<float, 0, TET4, 4> bf_vol (
      nullptr, bf_dynMem);
  edge::seismic::kernels::VolIntSingle<float, 0, TET4, 4> fp_vol (nullptr,
                                                                  fp_dynMem);

  float l_scratch[9][20][1];
  float l2_scratch[9][20][1];

  float result_bf[9][20][1];
  float result_fp[9][20][1];

  for (unsigned short l_qt = 0; l_qt < 9; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 20; l_md++)
        {
          result_fp[l_qt][l_md][0] = 0; //l_dofsE[l_qt][l_md];
          result_bf[l_qt][l_md][0] = 0; //l_dofsE[l_qt][l_md];
        }
    }

  // 20 = TL_N_MDS
  // 9 =  TL_N_QTS
  // starE = [9][9]
  // Dofs (degrees of freedom) = [9][20]
  // stiffness = [20][20]
  // cause its transposed ->
  // stiff * Dofs * starE
  // [20][20] * [20][9] * [9][9] = [20][9]

  // compute volume integration
  bf_vol.apply ((float (*)[81])l_starE, nullptr, nullptr,
                (float (*)[20][1])l_tDofsE, nullptr,
                (float (*)[20][1])result_bf, nullptr, l_scratch);

  fp_vol.apply ((float (*)[81])l_starE, nullptr, nullptr,
                (float (*)[20][1])l_tDofsE, nullptr,
                (float (*)[20][1])result_fp, nullptr, l2_scratch);

  // check the results
  for (unsigned short l_qt = 0; l_qt < 9; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 20; l_md++)
        {
           std::cout <<"result fp32 kernel:" << '\t'<<
           result_fp[l_qt][l_md][0] << std::endl;
          std::cout << "result bf16 kernel:" << '\t'
                    << result_bf[l_qt][l_md][0] << std::endl;
          // REQUIRE(result_fp[l_qt][l_md][0] ==
          // Approx(result_bf[l_qt][l_md][0]).epsilon(0.001));
          //std::cout << "reference value:" << '\t' << l_refEdofs[l_qt][l_md]
           //         << std::endl;
          // REQUIRE(result_bf[l_qt][l_md] ==
          //    Approx(l_refEdofs[l_qt][l_md]).epsilon(0.001));
          //      REQUIRE(l_dofsE[l_qt][l_md] ==
          //      l_refEdofs[l_qt][l_md]);
        }
    }
}

TEST_CASE ("Time tested bfloat against single precision kernel",
           "[elastic][VolIntSingle][bfloat][time]")
{
  // set up matrix structures
#include "VolInt.test.inc"

  // volume kernel
  edge::data::Dynamic bf_dynMem;
  edge::data::Dynamic fp_dynMem;
  edge::seismic::kernels::VolIntSingleBF16<float, 0, TET4, 4> bf_vol (
      nullptr, bf_dynMem);
  edge::seismic::kernels::VolIntSingle<float, 0, TET4, 4> fp_vol (nullptr,
                                                                  fp_dynMem);

  float l_scratch[9][20][1] = {};
  float input_dof[9][20][1] = {};
  for (size_t i = 0; i < 9; i++)
    {
      for (size_t j = 0; j < 20; j++)
        {
          input_dof[i][j][0] = 0;
        }
    }

  using namespace std::chrono;

  size_t count = 1000000;

  // compute volume integration
  auto start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      fp_vol.apply ((float (*)[81])l_starE, nullptr, nullptr,
                    (float (*)[20][1])input_dof, nullptr,
                    (float (*)[20][1])l_dofsE, nullptr, l_scratch);
    }
  auto end = system_clock::now ();

  double seconds = duration<double> (end - start).count ();

  std::cout << "Single precision kernel with " << count << " iterations took "
            << seconds << " seconds" << std::endl;

  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      bf_vol.apply ((float (*)[81])l_starE, nullptr, nullptr,
                    (float (*)[20][1])input_dof, nullptr,
                    (float (*)[20][1])l_dofsE, nullptr, l_scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count ();

  std::cout << "BF16 precision kernel with " << count << " iterations took "
            << seconds << " seconds" << std::endl;
}

TEST_CASE ("order 5", "[elastic][VolIntSingle][bfloat][order5]")
{
  // set up matrix structures
#include "VolInt.test.inc"

  // volume kernel
  edge::data::Dynamic bf_dynMem;
  edge::data::Dynamic fp_dynMem;

  edge::seismic::kernels::VolIntSingle<float, 0, TET4, 5> fp_vol (nullptr,
                                                                  fp_dynMem);

  edge::seismic::kernels::VolIntSingleBF16<float, 0, TET4, 5> bf_vol (
      nullptr, bf_dynMem);

  float l_scratch[9][35][1];

  float input_dof[9][35][1];
  float input_starE[9][9][1];

  float result_bf[9][35][1];
  float result_fp[9][35][1];

  for (size_t i = 0; i < 3; i++)
    {
      for (size_t j = 0; j < 9; j++)
        {
          for (size_t d = 0; d < 9; d++)
            {
              for (size_t l = 0; l < 35; l++)
                {
                  input_dof[d][l][0] = 1;
                  input_starE[i][d + j * d][0] = 1;
                }
            }
        }
    }

  for (size_t i = 0; i < 10000; i++)
    {
      bf_vol.apply ((float (*)[81])input_starE, nullptr, nullptr,
                    (float (*)[35][1])input_dof, nullptr,
                    (float (*)[35][1])result_bf, nullptr, l_scratch);
      fp_vol.apply ((float (*)[81])input_starE, nullptr, nullptr,
                    (float (*)[35][1])input_dof, nullptr,
                    (float (*)[35][1])result_fp, nullptr, l_scratch);
    }
  for (unsigned short l_qt = 0; l_qt < 9; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 35; l_md++)
        {
          std::cout << "result fp32 kernel:" << '\t'
                    << result_fp[l_qt][l_md][0] << std::endl;
          std::cout << "result bf16 kernel:" << '\t'
                    << result_bf[l_qt][l_md][0] << std::endl;
          // REQUIRE(result_fp[l_qt][l_md][0] ==
          // Approx(result_bf[l_qt][l_md][0]).epsilon(0.001));
          // std::cout <<"reference value:" << '\t'<< l_refEdofs[l_qt][l_md] <<
          // std::endl;
          // REQUIRE(result_bf[l_qt][l_md] ==
          //   Approx(l_refEdofs[l_qt][l_md]).epsilon(0.001));
          //     REQUIRE(l_dofsE[l_qt][l_md] ==
          //     l_refEdofs[l_qt][l_md]);
        }
    }
}
/*
TEST_CASE ("Time tested bfloat functions",
           "[elastic][VolIntSingle][bfloat][timers]")
{
  // set up matrix structures
#include "VolInt.test.inc"

  // volume kernel
  edge::data::Dynamic bf_dynMem;
  edge::seismic::kernels::VolIntSingleBF16<float, 0, TET4, 4> bf_vol (
      nullptr, bf_dynMem);

  float l_scratch[9][20][1];

  size_t count = 1000000;

  // compute volume integration

  for (size_t i = 0; i < count; i++)
    {
      bf_vol.apply ((float (*)[81])l_starE, nullptr, nullptr,
                    (float (*)[20][1])l_tDofsE, nullptr,
                    (float (*)[20][1])l_dofsE, nullptr, l_scratch);
    }

  std::cout << "Allocating memory costed: " << '\t' << bf_vol.sec_allocate
            << " seconds." << std::endl;
  std::cout << "Padding Dimensions in B costed: " << '\t'
            << bf_vol.sec_dimpadding << " seconds." << std::endl;
  std::cout << "Generate BF matrices costed: " << '\t' << bf_vol.sec_gen_mat
            << " seconds." << std::endl;
  std::cout << "Kernels took " << '\t' << bf_vol.sec_kernels << " seconds."
            << std::endl;
}
*/

TEST_CASE ("Time tested bfloat setup", "[elastic][VolIntSingle][bfloat][setup]")
{
  // set up matrix structures
#include "VolInt.test.inc"

  // volume kernel
  edge::data::Dynamic bf_dynMem;
  edge::seismic::kernels::VolIntSingleBF16<float, 0, TET4, 4> bf_vol (
      nullptr, bf_dynMem);

  float l_scratch[9][20][1] = {};
  float input_dof[9][20][1] = {};
  for (size_t i = 0; i < 9; i++)
    {
      for (size_t j = 0; j < 20; j++)
        {
          input_dof[i][j][0] = 0;
        }
    }

  using namespace std::chrono;

  size_t count = 1000000;

  auto start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      bf_vol.apply ((float (*)[81])l_starE, nullptr, nullptr,
                    (float (*)[20][1])input_dof, nullptr,
                    (float (*)[20][1])l_dofsE, nullptr, l_scratch);
    }
  auto end = system_clock::now ();

  double seconds = duration<double> (end - start).count ();

  std::cout << "BF16 setup Volint for  " << count << " iterations took "
            << seconds << " seconds" << std::endl;
}
TEST_CASE ("order 5 time setup", "[elastic][VolIntSingle][bfloat][order5]")
{
  // set up matrix structures
#include "VolInt.test.inc"

  // volume kernel
  edge::data::Dynamic bf_dynMem;
  edge::data::Dynamic fp_dynMem;

  edge::seismic::kernels::VolIntSingle<float, 0, TET4, 5> fp_vol (nullptr,
                                                                  fp_dynMem);

  edge::seismic::kernels::VolIntSingleBF16<float, 0, TET4, 5> bf_vol (
      nullptr, bf_dynMem);

  float l_scratch[9][35][1] = {};

  float input_dof[9][35][1] = {};
  float input_starE[9][9][1] = {};

  float result_bf[9][35][1] = {};
  float result_fp[9][35][1] = {};

  for (size_t i = 0; i < 3; i++)
    {
      for (size_t j = 0; j < 9; j++)
        {
          for (size_t d = 0; d < 9; d++)
            {
              for (size_t l = 0; l < 35; l++)
                {
                  input_dof[d][l][0] = 0;
                  input_starE[i][d + j * d][0] = 0;
                }
            }
        }
    }
    using namespace std::chrono;

  size_t count = 1000000;

  auto start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      bf_vol.apply ((float (*)[81])input_starE, nullptr, nullptr,
                    (float (*)[35][1])input_dof, nullptr,
                    (float (*)[35][1])result_bf, nullptr, l_scratch);
      
    }
    auto end = system_clock::now ();
    double seconds = duration<double> (end - start).count ();
    std::cout << "BF16 setup Volint for order 5 and   " << count << " iterations took "
            << seconds << " seconds" << std::endl;
     start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
    fp_vol.apply ((float (*)[81])input_starE, nullptr, nullptr,
                    (float (*)[35][1])input_dof, nullptr,
                    (float (*)[35][1])result_fp, nullptr, l_scratch);
    }
    end = system_clock::now ();
    seconds = duration<double> (end - start).count ();
    std::cout << "FP32 setup Volint for order 5 and   " << count << " iterations took "
            << seconds << " seconds" << std::endl;
}
TEST_CASE ("order 3 time setup", "[elastic][VolIntSingle][bfloat][order3]")
{
  // set up matrix structures
#include "VolInt.test.inc"

  // volume kernel
  edge::data::Dynamic bf_dynMem;
  edge::data::Dynamic fp_dynMem;

  edge::seismic::kernels::VolIntSingle<float, 0, TET4, 3> fp_vol (nullptr,
                                                                  fp_dynMem);

  edge::seismic::kernels::VolIntSingleBF16<float, 0, TET4, 3> bf_vol (
      nullptr, bf_dynMem);

  float l_scratch[9][10][1] = {};

  float input_dof[9][10][1] = {};
  float input_starE[9][9][1] = {};

  float result_bf[9][10][1] = {};
  float result_fp[9][10][1] = {};

  for (size_t i = 0; i < 3; i++)
    {
      for (size_t j = 0; j < 9; j++)
        {
          for (size_t d = 0; d < 9; d++)
            {
              for (size_t l = 0; l < 10; l++)
                {
                  input_dof[d][l][0] = 0;
                  input_starE[i][d + j * d][0] = 0;
                }
            }
        }
    }
    using namespace std::chrono;

  size_t count = 1000000;

  auto start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      bf_vol.apply ((float (*)[81])input_starE, nullptr, nullptr,
                    (float (*)[10][1])input_dof, nullptr,
                    (float (*)[10][1])result_bf, nullptr, l_scratch);
      
    }
    auto end = system_clock::now ();
    double seconds = duration<double> (end - start).count ();
    std::cout << "BF16 setup Volint for order 3 and   " << count << " iterations took "
            << seconds << " seconds" << std::endl;
     start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
    fp_vol.apply ((float (*)[81])input_starE, nullptr, nullptr,
                    (float (*)[10][1])input_dof, nullptr,
                    (float (*)[10][1])result_fp, nullptr, l_scratch);
    }
    end = system_clock::now ();
    seconds = duration<double> (end - start).count ();
    std::cout << "FP32 setup Volint for order 3 and   " << count << " iterations took "
            << seconds << " seconds" << std::endl;
}