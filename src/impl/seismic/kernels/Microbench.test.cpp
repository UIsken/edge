#include <catch.hpp>
#define private public
#include "data/MmXsmmSingle.hpp"
#include "VolIntSingleBF16.hpp"
#include <iostream>
#include <chrono>
#undef private

TEST_CASE ("Microbenching volume kernel mmkernels", "[microbenchkernels]") {
    edge::data::MmXsmmSingle<libxsmm_bfloat16> m_bf;
    edge::data::MmXsmmSingle<float> m_fp;
    double gflops;
    SECTION("Order 4"){
    m_bf.add(
        0,
        20,
        9,
        20,
        20,
        20, 
        20,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    m_bf.add(
        0,
        20,
        9,
        10,
        20,
        10, 
        20,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    m_fp.add(
        0,
        20,
        9,
        20,
        20,
        20, 
        20,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    m_fp.add(
        0,
        20,
        9,
        9,
        20,
        9, 
        20,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    libxsmm_bfloat16 star[9*10];
    float star_fp[9*10];
    for (size_t i = 0; i < 9*10; i++)
    {
        star[i]=0;
        star_fp[i]=0;
    }
    libxsmm_bfloat16 stiff[20*20];
    float stiff_fp[20*20];
    for (size_t i = 0; i < 20*20; i++)
    {
        stiff[i] =0;
        stiff_fp[i]=0;
    }
    libxsmm_bfloat16 DoFs[10*20];
    float DoFs_fp[10*20];
    float scratch[10*20];
    for (size_t i = 0; i < 10*20; i++)
    {
        DoFs[i] =0;
        DoFs_fp[i]=0;
        scratch[i] =0;
    }
    using namespace std::chrono;

  size_t count = 10000000;

  auto start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_bf.m_kernels[0][0](stiff, DoFs, scratch);
    }
  auto end = system_clock::now ();

  double seconds = duration<double> (end - start).count ();
  gflops = (double)2*(double)20*(double)20*(double)9*((double)count/(double)1e9);
  std::cout << "BF Kernel 20x20x9 and order 4 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;

  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_fp.m_kernels[0][0](stiff_fp, DoFs_fp, scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count ();

  std::cout << "FP Kernel 20x20x9 and order 4 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;

  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_bf.m_kernels[0][1](DoFs, star, scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count();
  gflops = (double)2*(double)20*(double)9*(double)9*((double)count/(double)1e9);
  std::cout << "BF Kernel 20x9x9 and order 4 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;
            
  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_fp.m_kernels[0][1](DoFs_fp, star_fp, scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count ();

  std::cout << "FP Kernel 20x9x9 and order 4 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;
    }
  SECTION("Order 5"){
    m_bf.add(
        0,
        35,
        9,
        36,
        35,
        36, 
        35,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    m_bf.add(
        0,
        35,
        9,
        10,
        35,
        10, 
        35,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    m_fp.add(
        0,
        35,
        9,
        35,
        35,
        35, 
        35,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    m_fp.add(
        0,
        35,
        9,
        9,
        35,
        9, 
        35,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    libxsmm_bfloat16 star[9*10];
    float star_fp[9*10];
    for (size_t i = 0; i < 9*10; i++)
    {
        star[i]=0;
        star_fp[i]=0;
    }
    libxsmm_bfloat16 stiff[35*36];
    float stiff_fp[35*36];
    for (size_t i = 0; i < 35*36; i++)
    {
        stiff[i] =0;
        stiff_fp[i]=0;
    }
    libxsmm_bfloat16 DoFs[9*36];
    float DoFs_fp[9*36];
    float scratch[9*36];
    for (size_t i = 0; i < 9*36; i++)
    {
        DoFs[i] =0;
        DoFs_fp[i]=0;
        scratch[i] =0;
    }
    using namespace std::chrono;

  size_t count = 10000000;

  auto start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_bf.m_kernels[0][0](stiff, DoFs, scratch);
    }
  auto end = system_clock::now ();

  double seconds = duration<double> (end - start).count ();
  gflops = (double)2*(double)35*(double)35*(double)9*((double)count/(double)1e9);
  std::cout << "BF Kernel 35x35x9 and order 5 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;

  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_fp.m_kernels[0][0](stiff_fp, DoFs_fp, scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count ();

  std::cout << "FP Kernel 35x35x9 and order 5 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;

  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_bf.m_kernels[0][1](DoFs, star, scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count();
  gflops = (double)2*(double)35*(double)9*(double)9*((double)count/(double)1e9);
  std::cout << "BF Kernel 35x9x9 and order 5 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;
            
  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_fp.m_kernels[0][1](DoFs_fp, star_fp, scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count ();

  std::cout << "FP Kernel 35x9x9 and order 5 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;
    }
  SECTION("Order 6"){
    m_bf.add(
        0,
        56,
        9,
        56,
        56,
        56, 
        56,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    m_bf.add(
        0,
        56,
        9,
        10,
        56,
        10, 
        56,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    m_fp.add(
        0,
        56,
        9,
        56,
        56,
        56, 
        56,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    m_fp.add(
        0,
        56,
        9,
        9,
        56,
        9, 
        56,
        1.0,
        1.0,
        LIBXSMM_GEMM_PREFETCH_NONE);
    
    libxsmm_bfloat16 star[9*10];
    float star_fp[9*10];
    for (size_t i = 0; i < 9*10; i++)
    {
        star[i]=0;
        star_fp[i]=0;
    }
    libxsmm_bfloat16 stiff[56*56];
    float stiff_fp[56*56];
    for (size_t i = 0; i < 56*56; i++)
    {
        stiff[i] =0;
        stiff_fp[i]=0;
    }
    libxsmm_bfloat16 DoFs[9*56];
    float DoFs_fp[9*56];
    float scratch[9*56];
    for (size_t i = 0; i < 9*56; i++)
    {
        DoFs[i] =0;
        DoFs_fp[i]=0;
        scratch[i] =0;
    }
    using namespace std::chrono;

  size_t count = 10000000;

  auto start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_bf.m_kernels[0][0](stiff, DoFs, scratch);
    }
  auto end = system_clock::now ();

  double seconds = duration<double> (end - start).count ();

  gflops = (double)2*(double)56*(double)56*(double)9*((double)count/(double)1e9);
  std::cout << "BF Kernel 56x56x9 and order 6 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;

  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_fp.m_kernels[0][0](stiff_fp, DoFs_fp, scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count ();

  std::cout << "FP Kernel 56x56x9 and order 6 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;

  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_bf.m_kernels[0][1](DoFs, star, scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count();
  
  gflops = (double)2*(double)56*(double)9*(double)9*((double)count/(double)1e9);
  std::cout << "BF Kernel 56x9x9 and order 6 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;
            
  start = system_clock::now ();
  for (size_t i = 0; i < count; i++)
    {
      m_fp.m_kernels[0][1](DoFs_fp, star_fp, scratch);
    }
  end = system_clock::now ();

  seconds = duration<double> (end - start).count ();

  std::cout << "FP Kernel 56x9x9 and order 6 and " << count << " iterations took "
            << seconds << " seconds and GFLOPS " << gflops/seconds << std::endl;
    }
}

TEST_CASE("Microbenching disjunction", "[microbenchkernels]"){

  // volume kernel
  edge::data::Dynamic l_dynMem;
  edge::seismic::kernels::VolIntSingleBF16<float, 0, TET4, 4> l_vol (nullptr,
                                                                     l_dynMem);
  size_t count = 10000000;
  SECTION("generateing matrices"){

      
    float star[4000];
    float starx[4000];
    libxsmm_bfloat16 star1[4000];
    libxsmm_bfloat16 star2[4000];
    libxsmm_bfloat16 star3[4000];

    int M[3] = {20, 35, 56};
    for (size_t i = 0; i < 4000; i++)
    {
        star[i]=i * 0.1;
        star1[i]=0;
        star2[i]=0;
        star3[i]=0;
    }

    using namespace std::chrono;


    auto start = system_clock::now ();
    for (size_t i = 0; i < count; i++)
    {
      
      l_vol.gen_bf_matrices<9 * 9> (star, star1, star2, star3);
    }
    auto end = system_clock::now ();
    double seconds = duration<double> (end - start).count ();

    std::cout <<"generating 9x9 naive took " << seconds << " for "<< count<< " iterations" << std::endl;

    start = system_clock::now ();
    for (size_t i = 0; i < count; i++)
    {
      
      l_vol.split_compress(star, star1, star2, star3, 9*9);
    }
    end = system_clock::now ();
    seconds = duration<double> (end - start).count ();

    std::cout <<"generating 9x9 vectorized took " << seconds << " for "<< count<< " iterations" << std::endl;
    start = system_clock::now ();
    
  
    start = system_clock::now ();
    for (size_t j = 0; j < count; j++)
    {
      
      l_vol.gen_bf_matrices<20 * 20 > (star, star1, star2, star3);
    }
    end = system_clock::now ();
    seconds = duration<double> (end - start).count ();

    std::cout <<"generating 20x20 naive took " << seconds << " for "<< count<< " iterations" << std::endl;

    start = system_clock::now ();
    for (size_t j = 0; j < count; j++)
    {
      
      l_vol.split_compress(star, star1, star2, star3, 20*20);
    }
    end = system_clock::now ();
    seconds = duration<double> (end - start).count ();

    std::cout <<"generating 20x20 vectorized took " << seconds << " for "<< count<< " iterations" << std::endl;
    start = system_clock::now ();
    start = system_clock::now ();
    for (size_t j = 0; j < count; j++)
    {
      
      l_vol.gen_bf_matrices<20*9> (star, star1, star2, star3);
    }
    end = system_clock::now ();
    seconds = duration<double> (end - start).count ();

    std::cout <<"generating 20x9 naive took " << seconds << " for "<< count<< " iterations" << std::endl;

    start = system_clock::now ();
    for (size_t j = 0; j < count; j++)
    {
      
      l_vol.split_compress(star, star1, star2, star3, 20*9);
    }
    end = system_clock::now ();
    seconds = duration<double> (end - start).count ();

    std::cout <<"generating 20x9 vectorized took " << seconds << " for "<< count<< " iterations" << std::endl;
    start = system_clock::now ();
 
    
  }
}