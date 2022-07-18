#include "api.h"
#include "pal.h"
#include "pal_regression.h"

int main(int argc, char** argv) {
    int count = 0;

    pal_printf("In process: %s", argv[0]);
    for (int i = 1; i < argc; i++) {
        pal_printf(" %s", argv[i]);
    }
    pal_printf("\n");

    if (argc == 1) {
        PAL_HANDLE pipe_srv = NULL;
        int ret = DkStreamOpen("pipe.srv:Process4", PAL_ACCESS_RDWR, /*share_flags=*/0,
                               PAL_CREATE_IGNORED, /*options=*/0, &pipe_srv);
        if (ret < 0) {
            pal_printf("DkStreamOpen(\"pipe.srv\", ...) failed: %d\n", ret);
            return 1;
        }

        uint64_t time = 0;
        if (DkSystemTimeQuery(&time) < 0) {
            pal_printf("DkSystemTimeQuery failed\n");
            return 1;
        }
        char time_arg[24];
        snprintf(time_arg, 24, "%ld", time);

        const char* newargs[4] = {"Process4", "0", time_arg, NULL};

        PAL_HANDLE proc = NULL;
        ret = DkProcessCreate(newargs, &proc);

        if (ret < 0)
            pal_printf("Can't create process\n");

        DkObjectClose(proc);

        PAL_HANDLE pipe = NULL;
        ret = DkStreamWaitForClient(pipe_srv, &pipe, /*options=*/0);
        if (ret < 0) {
            pal_printf("DkStreamWaitForClient failed: %d\n", ret);
        }
        DkObjectClose(pipe);
        DkObjectClose(pipe_srv);
    } else {
        count = atoi(argv[1]);

        if (count < 100) {
            count++;

            char count_arg[12];
            snprintf(count_arg, 12, "%d", count);
            const char* newargs[4] = {"Process4", count_arg, argv[2], NULL};

            PAL_HANDLE proc = NULL;
            int ret = DkProcessCreate(newargs, &proc);

            if (ret < 0)
                pal_printf("Can't create process\n");

            DkObjectClose(proc);
        } else {
            uint64_t end = 0;
            if (DkSystemTimeQuery(&end) < 0) {
                pal_printf("DkSystemTimeQuery failed\n");
                return 1;
            }
            uint64_t start = atol(argv[2]);
            pal_printf("wall time = %ld\n", end - start);

            PAL_HANDLE pipe = NULL;
            int ret = DkStreamOpen("pipe:Process4", PAL_ACCESS_RDWR, /*share_flags=*/0,
                                   PAL_CREATE_IGNORED, /*options=*/0, &pipe);
            if (ret < 0) {
                pal_printf("Failed to open pipe: %d\n", ret);
                return 1;
            }
            DkObjectClose(pipe);
        }
    }

    DkProcessExit(0);
}
