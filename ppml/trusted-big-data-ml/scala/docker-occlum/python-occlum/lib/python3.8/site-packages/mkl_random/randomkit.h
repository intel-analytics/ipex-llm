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
#include "mkl_vsl.h"
#include "Python.h"
#include "numpy/npy_common.h"

#ifndef _I_RANDOMKIT_
#define _I_RANDOMKIT_

typedef struct irk_state_
{
    VSLStreamStatePtr stream;
} irk_state;

typedef enum {
    RK_NOERR = 0, /* no error */
    RK_ENODEV = 1, /* no RK_DEV_RANDOM device */
    RK_ERR_MAX = 2
} irk_error;

/* if changing this, also adjust brng_list[BRNG_KINDS] in randomkit.c */
#define BRNG_KINDS 11

typedef enum {
    MT19937       = 0,
    SFMT19937     = 1,
    WH            = 2,
    MT2203        = 3,
    MCG31         = 4,
    R250          = 5,
    MRG32K3A      = 6,
    MCG59         = 7,
    PHILOX4X32X10 = 8,
    NONDETERM     = 9,
    ARS5          = 10
} irk_brng_t;


/* error strings */
extern char *irk_strerror[RK_ERR_MAX];

/* Maximum generated random value */
#define RK_MAX 0xFFFFFFFFUL

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the RNG state using the given seed.
 */


/*
 * Initialize the RNG state using a random seed.
 * Uses /dev/random or, when unavailable, the clock (see randomkit.c).
 * Returns RK_NOERR when no errors occurs.
 * Returns RK_ENODEV when the use of RK_DEV_RANDOM failed (for example because
 * there is no such device). In this case, the RNG was initialized using the
 * clock.
 */

/*
 * Initialize the RNG state using the given seed.
 */
extern void irk_dealloc_stream(irk_state *state);
extern void irk_seed_mkl(irk_state *state, const unsigned int seed, const irk_brng_t brng, const unsigned int stream_id);
extern void irk_seed_mkl_array(irk_state *state, const unsigned int *seed_vec,
    const int seed_len, const irk_brng_t brng, const unsigned int stream_id);
extern irk_error irk_randomseed_mkl(irk_state *state, const irk_brng_t brng, const unsigned int stream_id);
extern int irk_get_stream_size(irk_state *state);
extern void irk_get_state_mkl(irk_state *state, char * buf);
extern int irk_set_state_mkl(irk_state *state, char * buf);
extern int irk_get_brng_mkl(irk_state *state);
extern int irk_get_brng_and_stream_mkl(irk_state *state, unsigned int* stream_id);

extern int irk_leapfrog_stream_mkl(irk_state *state, const int k, const int nstreams);
extern int irk_skipahead_stream_mkl(irk_state *state, const long long int nskip);

/*
 * fill the buffer with size random bytes
 */
extern void irk_fill(void *buffer, size_t size, irk_state *state);

/*
 * fill the buffer with randombytes from the random device
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 * On Unix, if strong is defined, RK_DEV_RANDOM is used. If not, RK_DEV_URANDOM
 * is used instead. This parameter has no effect on Windows.
 * Warning: on most unixes RK_DEV_RANDOM will wait for enough entropy to answer
 * which can take a very long time on quiet systems.
 */
extern irk_error irk_devfill(void *buffer, size_t size, int strong);

/*
 * fill the buffer using irk_devfill if the random device is available and using
 * irk_fill if is is not
 * parameters have the same meaning as irk_fill and irk_devfill
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 */
extern irk_error irk_altfill(void *buffer, size_t size, int strong,
                            irk_state *state);

#ifdef __cplusplus
}
#endif

#endif /* _I_RANDOMKIT_ */
