/* use `-fno-builtin` compiler option to prevent nearbyint and fesetround being optimized out */

#include <err.h>
#include <fenv.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static int rounding_modes[] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};

static const char* get_rounding_mode(int mode) {
    switch (mode) {
        case FE_TONEAREST:  return "FE_TONEAREST ";
        case FE_UPWARD:     return "FE_UPWARD    ";
        case FE_DOWNWARD:   return "FE_DOWNWARD  ";
        case FE_TOWARDZERO: return "FE_TOWARDZERO";
    }
    return "UNKNOWN";
}

static void* thread_fp(void* arg) {
    int mode = fegetround();
    printf("%s  child: 42.5 = %.1f, -42.5 = %.1f\n", get_rounding_mode(mode),
           nearbyint(42.5), nearbyint(-42.5));
    return arg;
}

int main(int argc, char* argv[]) {
    int ret;

    for (unsigned int i = 0; i < sizeof(rounding_modes)/sizeof(rounding_modes[0]); i++) {
        int mode = rounding_modes[i];
        ret = fesetround(mode);
        if (ret)
            err(EXIT_FAILURE, "fesetround failed");

        pthread_t thread;
        ret = pthread_create(&thread, NULL, thread_fp, NULL);
        if (ret)
            err(EXIT_FAILURE, "pthread_create failed");

        ret = pthread_join(thread, NULL);
        if (ret)
            err(EXIT_FAILURE, "pthread_join failed");

        printf("%s parent: 42.5 = %.1f, -42.5 = %.1f\n", get_rounding_mode(mode),
               nearbyint(42.5), nearbyint(-42.5));
    }

    return 0;
}
