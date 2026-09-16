#ifndef NN_RANDOM_H
#define NN_RANDOM_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "common.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bcrypt.h>
#endif

typedef struct nn_rng {
    uint64_t state;
} nn_rng;

/* SplitMix64: a fixed integer sequence, independent of libc rand/distributions. */
static NN_FORCEINLINE uint32_t nn_rng_next(nn_rng *rng)
{
    uint64_t z = (rng->state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return (uint32_t)((z ^ (z >> 31)) >> 32);
}

static inline uint32_t nn_random_seed(void)
{
    uint32_t seed = 0;
#if defined(_WIN32)
    if (BCryptGenRandom(NULL, (PUCHAR)&seed, sizeof(seed),
            BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0 && seed != 0)
        return seed;
#else
    FILE *stream = fopen("/dev/urandom", "rb");
    if (stream != NULL) {
        const size_t count = fread(&seed, sizeof(seed), 1, stream);
        fclose(stream);
        if (count == 1 && seed != 0) return seed;
    }
#endif
    {
        struct timespec now;
        nn_rng fallback;
        now.tv_sec = time(NULL);
        now.tv_nsec = 0;
        (void)timespec_get(&now, TIME_UTC);
        fallback.state = (uint64_t)now.tv_sec ^ ((uint64_t)now.tv_nsec << 32)
            ^ (uint64_t)(uintptr_t)&seed ^ (uint64_t)clock();
        seed = nn_rng_next(&fallback);
    }
    return seed != 0 ? seed : UINT32_C(1);
}

static inline void nn_rng_init(nn_rng *rng, uint32_t seed)
{
    rng->state = seed != 0 ? seed : nn_random_seed();
}

/* Rejection removes modulo bias. A zero bound denotes an empty range. */
static inline uint32_t nn_rng_bounded(nn_rng *rng, uint32_t bound)
{
    uint32_t value, threshold;
    if (bound == 0) return 0;
    threshold = (uint32_t)(0u - bound) % bound;
    do { value = nn_rng_next(rng); } while (value < threshold);
    return value % bound;
}
#endif
