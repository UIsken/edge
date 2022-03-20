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
 * Tests the surface integration for single seismic simulations.
 **/
#include <catch.hpp>
#define private public
#include "SurfIntSingle.hpp"
#include "SurfIntSingleBF16.hpp"
#undef private

TEST_CASE ("Local surface integration for single seismic simulations with "
           "bfloat approxmation.",
           "[elastic][SurfIntLocalSingle][bfloat][SurfInt]")
{
  // set up matrix structures
#include "SurfInt.test.inc"

  // kernel
  edge::data::Dynamic l_dynMem;
  edge::seismic::kernels::SurfIntSingleBF16<float, 0, TET4, 3> l_surf (
      nullptr, l_dynMem);

  float l_scratch[2][9][6][1];

  // compute local surface integration
  l_surf.local ((float (*)[81])l_fSolvE, nullptr, (float (*)[10][1])l_tDofsE,
                (float (*)[10][1])l_dofsE, nullptr, l_scratch);

  // check the results
  for (unsigned short l_qt = 0; l_qt < 9; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 10; l_md++)
        {
          std::cout << "result bf16 kernel:" << '\t' << l_dofsE[l_qt][l_md]
                    << std::endl;
          std::cout << "reference value:" << '\t' << l_refEdofs[l_qt][l_md]
                    << std::endl;
          REQUIRE (l_dofsE[l_qt][l_md]
                   == Approx (l_refEdofs[l_qt][l_md]).epsilon (0.01));
        }
    }
}

TEST_CASE ("Neighboring surface integration for single seismic forward "
           "simulations with bfloat approxmation.",
           "[elastic][SurfIntNeighSingle][bfloat][SurfInt]")
{
  // set up matrix structures
#include "SurfInt.test.inc"

  // kernel
  edge::data::Dynamic l_dynMem;
  edge::seismic::kernels::SurfIntSingleBF16<float, 0, TET4, 3> l_surf (
      nullptr, l_dynMem);

  float l_scratch[2][9][6][1];

  // compute local surface integration
  l_surf.neigh (3, 1, 2, (float (*))l_fSolvE[0], nullptr,
                (float (*)[10][1])l_tDofsE, nullptr, (float (*)[10][1])l_dofsE,
                nullptr, l_scratch);

  // check the results
  for (unsigned short l_qt = 0; l_qt < 9; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 10; l_md++)
        {
          std::cout << "result bf16 kernel:" << '\t' << l_dofsE[l_qt][l_md]
                    << std::endl;
          std::cout << "reference value:" << '\t'
                    << l_refEneighDofs[l_qt][l_md] << std::endl;
          REQUIRE (l_dofsE[l_qt][l_md]
                   == Approx (l_refEneighDofs[l_qt][l_md]).epsilon (0.01));
        }
    }
}

TEST_CASE ("Neighboring surface integration in the presence of a free surface "
           "for single seismic forward simulations with bfloat approxmation.",
           "[elastic][SurfIntNeighFsSingle][bfloat][SurfInt]")
{
  // set up matrix structures
#include "SurfInt.test.inc"

  // kernel
  edge::data::Dynamic l_dynMem;
  edge::seismic::kernels::SurfIntSingleBF16<float, 0, TET4, 3> l_surf (
      nullptr, l_dynMem);

  float l_scratch[2][9][6][1];

  // compute local surface integration
  l_surf.neigh (2, std::numeric_limits<unsigned short>::max (),
                std::numeric_limits<unsigned short>::max (),
                (float (*))l_fSolvE[0], nullptr, (float (*)[10][1])l_tDofsE,
                nullptr, (float (*)[10][1])l_dofsE, nullptr, l_scratch);

  // check the results
  for (unsigned short l_qt = 0; l_qt < 9; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 10; l_md++)
        {
          std::cout << "result bf16 kernel:" << '\t' << l_dofsE[l_qt][l_md]
                    << std::endl;
          std::cout << "reference value:" << '\t'
                    << l_refEneighFdofs[l_qt][l_md] << std::endl;
          REQUIRE (l_dofsE[l_qt][l_md]
                   == Approx (l_refEneighFdofs[l_qt][l_md]).epsilon (0.01));
        }
    }
}

TEST_CASE ("Viscoelastic local surface integration for single forward "
           "simulations with bfloat approxmation.",
           "[visco][SurfIntLocalSingle][bfloat][SurfInt]")
{
  // set up matrix structures
#include "SurfInt.test.inc"

  // set up kernel
  edge::data::Dynamic l_dynMem;

  edge::seismic::kernels::SurfIntSingleBF16<float, 3, TET4, 3> l_surf (
      l_rfs, l_dynMem);

  float l_scratch[2][9][6][1];

  // compute local surface integration
  l_surf.local ((float (*)[81])l_fSolvE, (float (*)[6 * 9]) l_fSolvA,
                (float (*)[10][1])l_tDofsE, (float (*)[10][1])l_dofsE,
                (float (*)[6][10][1])l_dofsA, l_scratch);

  // check the results
  for (unsigned short l_qt = 0; l_qt < 9; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 10; l_md++)
        {
          REQUIRE (l_dofsE[l_qt][l_md]
                   == Approx (l_refEdofs[l_qt][l_md]).epsilon (0.01));
        }
    }

  for (unsigned short l_rm = 0; l_rm < 3; l_rm++)
    {
      for (unsigned short l_qt = 0; l_qt < 6; l_qt++)
        {
          for (unsigned short l_md = 0; l_md < 10; l_md++)
            {
              REQUIRE (l_dofsA[l_rm][l_qt][l_md]
                       == Approx (l_refVdofsA[l_rm][l_qt][l_md]));
            }
        }
    }
}

TEST_CASE (
    "Viscoelastic neighboring surface integration in the presence of a "
    "free surface for single forward simulations with bfloat approxmation.",
    "[visco][SurfIntNeighFsSingle][bfloat][SurfInt]")
{
  // set up matrix structures
#include "SurfInt.test.inc"

  // kernel
  edge::data::Dynamic l_dynMem;
  edge::seismic::kernels::SurfIntSingleBF16<float, 3, TET4, 3> l_surf (
      l_rfs, l_dynMem);

  float l_scratch[2][9][6][1];
  float l_upsA[6][10];
  for (unsigned short l_qt = 0; l_qt < 6; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 10; l_md++)
        {
          l_upsA[l_qt][l_md] = 0;
        }
    }

  // compute local surface integration
  l_surf.neigh (2, std::numeric_limits<unsigned short>::max (),
                std::numeric_limits<unsigned short>::max (),
                (float (*))l_fSolvE[0], l_fSolvA[0][0],
                (float (*)[10][1])l_tDofsE, nullptr, (float (*)[10][1])l_dofsE,
                (float (*)[10][1])l_upsA, l_scratch);

  // check the elastic part of the results
  for (unsigned short l_qt = 0; l_qt < 9; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 10; l_md++)
        {
          REQUIRE (l_dofsE[l_qt][l_md]
                   == Approx (l_refEneighFdofs[l_qt][l_md]).epsilon (0.01));
        }
    }

  // check the anelastic part of the results
  for (unsigned short l_qt = 0; l_qt < 6; l_qt++)
    {
      for (unsigned short l_md = 0; l_md < 10; l_md++)
        {
          REQUIRE (l_upsA[l_qt][l_md]
                   == Approx (l_refVneighFdofsA[l_qt][l_md]).epsilon (0.01));
        }
    }
}
