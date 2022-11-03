/*
 Copyright (c) 2017-2019, Intel Corporation

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of Intel Corporation nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stddef.h>
#include "randomkit.h"

#ifndef _MKL_DISTRIBUTIONS_H_
#define _MKL_DISTRIBUTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MATRIX = 0,
    PACKED = 1,
    DIAGONAL = 2
} ch_st_enum;

extern void irk_double_vec(irk_state *state, npy_intp len, double *res);
extern void irk_uniform_vec(irk_state *state, npy_intp len, double *res, const double low, const double high);
extern void irk_standard_normal_vec_ICDF(irk_state *state, npy_intp len, double *res);
extern void irk_standard_normal_vec_BM1(irk_state *state, npy_intp len, double *res);
extern void irk_standard_normal_vec_BM2(irk_state *state, npy_intp len, double *res);

extern void irk_normal_vec_ICDF(irk_state *state, npy_intp len, double *res, const double loc, const double scale);
extern void irk_normal_vec_BM1(irk_state *state, npy_intp len, double *res, const double loc, const double scale);
extern void irk_normal_vec_BM2(irk_state *state, npy_intp len, double *res, const double loc, const double scale);

extern void irk_standard_t_vec(irk_state *state, npy_intp len, double *res, const double df);
extern void irk_chisquare_vec(irk_state *state, npy_intp len, double *res, const double df);

extern void irk_standard_exponential_vec(irk_state *state, npy_intp len, double *res);
extern void irk_standard_cauchy_vec(irk_state *state, npy_intp len, double *res);

extern void irk_standard_gamma_vec(irk_state *state, npy_intp len, double *res, const double shape);
extern void irk_exponential_vec(irk_state *state, npy_intp len, double *res, const double scale);
extern void irk_gamma_vec(irk_state *state, npy_intp len, double *res, const double shape, const double scale);

extern void irk_pareto_vec(irk_state *state, npy_intp len, double *res, const double alph);
extern void irk_power_vec(irk_state *state, npy_intp len, double *res, const double alph);

extern void irk_weibull_vec(irk_state *state, npy_intp len, double *res, const double alph);

extern void irk_rayleigh_vec(irk_state *state, npy_intp len, double *res, const double scale);

extern void irk_beta_vec(irk_state *state, npy_intp len, double *res, const double p, const double q);
extern void irk_f_vec(irk_state *state, npy_intp len, double *res, const double df_num, const double df_den);

extern void irk_noncentral_chisquare_vec(irk_state *state, npy_intp len, double *res, const double df, const double nonc);

extern void irk_laplace_vec(irk_state *vec, npy_intp len, double *res, const double loc, const double scale);
extern void irk_gumbel_vec(irk_state *vec, npy_intp len, double *res, const double loc, const double scale);
extern void irk_logistic_vec(irk_state *vec, npy_intp len, double *res, const double loc, const double scale);

extern void irk_lognormal_vec_ICDF(irk_state *state, npy_intp len, double *res, const double mean, const double sigma);
extern void irk_lognormal_vec_BM(irk_state *state, npy_intp len, double *res, const double mean, const double sigma);

extern void irk_wald_vec(irk_state *state, npy_intp len, double *res, const double mean, const double scale);

extern void irk_vonmises_vec(irk_state *state, npy_intp len, double *res, const double mu, const double kappa);

extern void irk_noncentral_f_vec(irk_state *state, npy_intp len, double *res, double df_num, double df_den, double nonc);
extern void irk_triangular_vec(irk_state *state, npy_intp len, double *res, double left, double mode, double right);

extern void irk_binomial_vec(irk_state *state, npy_intp len, int *res, const int n, const double p);

extern void irk_multinomial_vec(irk_state *state, npy_intp len, int *res, const int n, const int k, const double* pvec);

extern void irk_geometric_vec(irk_state *state, npy_intp len, int *res, const double p);
extern void irk_negbinomial_vec(irk_state *state, npy_intp len, int *res, const double a, const double p);
extern void irk_hypergeometric_vec(irk_state *state, npy_intp len, int *res, const int ls, const int ss, const int ms);
extern void irk_poisson_vec_PTPE(irk_state *state, npy_intp len, int *res, const double lambda);
extern void irk_poisson_vec_POISNORM(irk_state *state, npy_intp len, int *res, const double lambda);

extern void irk_poisson_vec_V(irk_state *state, npy_intp len, int *res, double *lambdas);

extern void irk_zipf_long_vec(irk_state *state, npy_intp len, long *res, const double alp);

extern void irk_logseries_vec(irk_state *state, npy_intp len, int *res, const double alp);

extern void irk_discrete_uniform_vec(irk_state *state, npy_intp len, int *res, const int low, const int high);

extern void irk_discrete_uniform_long_vec(irk_state *state, npy_intp len, long *res, const long low, const long high);

extern void irk_rand_int64_vec(irk_state *state, npy_intp len, npy_int64 *res, const npy_int64 lo, const npy_int64 hi);
extern void irk_rand_uint64_vec(irk_state *state, npy_intp len, npy_uint64 *res, const npy_uint64 lo, const npy_uint64 hi);
extern void irk_rand_int32_vec(irk_state *state, npy_intp len, npy_int32 *res, const npy_int32 lo, const npy_int32 hi);
extern void irk_rand_uint32_vec(irk_state *state, npy_intp len, npy_uint32 *res, const npy_uint32 lo, const npy_uint32 hi);
extern void irk_rand_int16_vec(irk_state *state, npy_intp len, npy_int16 *res, const npy_int16 lo, const npy_int16 hi);
extern void irk_rand_uint16_vec(irk_state *state, npy_intp len, npy_uint16 *res, const npy_uint16 lo, const npy_uint16 hi);
extern void irk_rand_int8_vec(irk_state *state, npy_intp len, npy_int8 *res, const npy_int8 lo, const npy_int8 hi);
extern void irk_rand_uint8_vec(irk_state *state, npy_intp len, npy_uint8 *res, const npy_uint8 lo, const npy_uint8 hi);
extern void irk_rand_bool_vec(irk_state *state, npy_intp len, npy_bool *res, const npy_bool lo, const npy_bool hi);

extern void irk_ulong_vec(irk_state *state, npy_intp len, unsigned long *res);
extern void irk_long_vec(irk_state *state, npy_intp len, long *res);

extern void irk_multinormal_vec_ICDF(irk_state *state, npy_intp len, double *res, const int dim,
    double *mean_vec, double *ch, const ch_st_enum storage_mode);

extern void irk_multinormal_vec_BM1(irk_state *state, npy_intp len, double *res, const int dim,
    double *mean_vec, double *ch, const ch_st_enum storage_mode);

extern void irk_multinormal_vec_BM2(irk_state *state, npy_intp len, double *res, const int dim,
    double *mean_vec, double *ch, const ch_st_enum storage_mode);

#ifdef __cplusplus
}
#endif


#endif
