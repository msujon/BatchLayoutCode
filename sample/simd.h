/*
 * NOTE: Majedul: The purpose of this header file is to hide all architecture 
 * dependent implementation of intrinsic vector codes 
 */
#ifndef _SIMD_H_
#define _SIMD_H_

#ifdef BLC_X86
   #if defined(BLC_AVXZ) || defined(BLC_AVX512)
      #if VALUETPYE == double
         #define VLEN 8
/*
 *       AVX512 double precision 
 */
         #define BCL_VTYPE __m512d 
         #define BCL_vuld(v_, p_) v_ = _mm512_loadu_pd(p_) 
         #define BCL_vld(v_, p_) v_ = _mm512_load_pd(p_) 
         #define BCL_vzero(v_) v_ = _mm512_setzero_pd() 
         #define BCL_vust(v_, p_) _mm512_storeu_pd(p_, v_) 
         #define BCL_vst(v_, p_)  _mm512_store_pd(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm512_set1_pd(*(p_))
         #define BCL_vadd(d_, s1_, s2_) d_ = _m512_add_pd(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _m512_sub_pd(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _m512_mul_pd(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _m512_div_pd(s1_, s2_)
         #define BCL_vmac(d_, s1_, s2_) d_ = _m512_fmadd_pd(s1_, s2_, d_)
         #define BCL_vrcp(d_) d_ = _mm512_rcp14_pd(d_); // reciprocal 
         #define BCL_maskz_vrcp(k_, d_) d_ = _mm512_rcp14_pd(k_, d_); // reciprocal 
         #define BCL_cvtint2mask(k_, ik) k_ = _cvtu32_mask8(ik_) 
      #elif VALUETYPE == float
         #define VLEN 16
         #define BCL_VTYPE __m512 
         #define BCL_vldu(v_, p_) v_ = _mm512_loadu_ps(p_) 
         #define BCL_vld(v_, p_) v_ = _mm512_load_ps(p_) 
         #define BCL_vzero(v_) v_ = _mm512_setzero_ps() 
         #define BCL_vstu(v_, p_) _mm512_storeu_ps(p_, v_) 
         #define BCL_vst(v_, p_)  _mm512_store_ps(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm512_set1_ps(*(p_))
         #define BCL_vadd(d_, s1_, s2_) d_ = _m512_add_ps(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _m512_sub_ps(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _m512_mul_ps(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _m512_div_ps(s1_, s2_)
         #define BCL_vmac(d_, s1_, s2_) d_ = _m512_fmadd_ps(s1_, s2_, d_)
         #define BCL_vrcp(d_) d_ = _mm512_rcp14_ps(d_); // reciprocal 
         #define BCL_maskz_vrcp(k_, d_) d_ = _mm512_rcp14_ps(k_, d_); // reciprocal 
         #define BCL_cvtint2mask(k_, ik) k_ = _cvtu32_mask8(ik_) 
      #else
         #error "Unsupported Value Type!"
      #endif
 /*
  *   inst format: inst(dist, src1, src2)
  */
   #elif defined(BLC_AVX2) || defined(BLC_AVXMAC) || defined(BLC_AVX) 
      #if VALUETPYE == double
         #define VLEN 4
      #elif VALUETYPE == float
         #define VLEN 8
      #else
         #error "Unsupported Value Type!"
   #elif defined(BLC_SSE2) || defined(BLC_SSE3)
      #if VALUETPYE == double
         #define VLEN 2
      #elif VALUETYPE == float
         #define VLEN 4
      #else
         #error "Unsupported Value Type!"
   #elif defined(BLC_SSE1)
      #if VALUETYPE == float
         #define VLEN 4
      #else // double not supported 
         #error "Unsupported Value Type!"
   #elif defined(BLC_SSE1)
   #else
      #error "Unsupported X86 SIMD!"
   #endif

#elif defined(BLC_VSX)  // openPower vector unit  

#elif defined(BLC_ARM64) // arm64 machine 

#elif defined(BLC_FRCGNUVEC) // GNUVEC by GCC  

#else
   #error "Unsupported Architecture!"
#endif


#endif
