#include "api.h"
#include "pal.h"
#include "pal_regression.h"

#define PORT   8000
#define NTRIES 10

char addr[40];
char time_arg[24];
char buffer[12];

int main(int argc, char** argv) {
    int i;
    int ret;

    if (argc == 1) {
        unsigned long time = 0;
        if (DkSystemTimeQuery(&time) < 0) {
            pal_printf("DkSystemTimeQuery failed\n");
            return 1;
        }
        pal_printf("start time = %lu\n", time);

        snprintf(time_arg, 24, "%ld", time);

        const char* newargs[4] = {"Tcp", time_arg, NULL};

        PAL_HANDLE srv = NULL;
        ret = DkStreamOpen("tcp.srv:127.0.0.1:8000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                           PAL_CREATE_IGNORED, /*options=*/0, &srv);

        if (ret < 0) {
            pal_printf("not able to create server\n");
            return 1;
        }

        ret = DkStreamGetName(srv, addr, sizeof(addr));
        if (ret < 0) {
            pal_printf("DkStreamGetName failed\n");
            return 1;
        }
        pal_printf("server bound on %s\n", addr);

        PAL_HANDLE proc = NULL;
        if (DkProcessCreate(newargs, &proc) < 0) {
            pal_printf("Tcp: DkProcessCreate failed\n");
            return 1;
        }

        for (i = 0; i < NTRIES; i++) {
            PAL_HANDLE cli = NULL;
            ret = DkStreamWaitForClient(srv, &cli, /*options=*/0);

            if (ret < 0) {
                pal_printf("not able to accept client\n");
                return 1;
            }

            ret = DkStreamGetName(cli, addr, sizeof(addr));
            if (ret < 0) {
                pal_printf("DkStreamGetName failed\n");
                return 1;
            }
            pal_printf("client accepted on %s\n", addr);

            char buf[12] = "Hello World";
            size_t buf_size = sizeof(buf);
            ret = DkStreamWrite(cli, 0, &buf_size, buf, NULL);

            if (ret < 0 || buf_size != sizeof(buf)) {
                pal_printf("not able to send to client\n");
                return 1;
            }

            DkObjectClose(cli);
        }

        int retval;
        size_t retval_size = sizeof(retval);
        ret = DkStreamRead(proc, 0, &retval_size, &retval, NULL, 0);
        if (ret < 0) {
            pal_printf("DkStreamRead failed\n");
            return 1;
        }
        if (retval_size != sizeof(retval)) {
            pal_printf("DkStreamRead - short read\n");
            return 1;
        }

        ret = DkStreamDelete(srv, PAL_DELETE_ALL);
        if (ret < 0) {
            pal_printf("DkStreamDelete failed\n");
            return 1;
        }
        DkObjectClose(srv);
    } else {
        for (i = 0; i < NTRIES; i++) {
            PAL_HANDLE cli = NULL;
            ret = DkStreamOpen("tcp:127.0.0.1:8000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                               PAL_CREATE_IGNORED, /*options=*/0, &cli);

            if (ret < 0) {
                pal_printf("not able to create client\n");
                return 1;
            }

            ret = DkStreamGetName(cli, addr, sizeof(addr));
            if (ret < 0) {
                pal_printf("DkStreamGetName failed\n");
                return 1;
            }
            pal_printf("client connected on %s\n", addr);

            size_t bytes = sizeof(buffer);
            ret = DkStreamRead(cli, 0, &bytes, buffer, NULL, 0);

            if (ret < 0 || bytes == 0) {
                pal_printf("not able to receive from server\n");
                return 1;
            }

            pal_printf("read from server: %s\n", buffer);

            ret = DkStreamDelete(cli, PAL_DELETE_ALL);
            if (ret < 0) {
                pal_printf("DkStreamDelete failed\n");
                return 1;
            }
            DkObjectClose(cli);
        }

        unsigned long end = 0;
        if (DkSystemTimeQuery(&end) < 0) {
            pal_printf("DkSystemTimeQuery failed\n");
            return 1;
        }
        pal_printf("end time = %lu\n", end);

        unsigned long start = atol(argv[1]);
        pal_printf("wall time = %ld\n", end - start);

        int retval = 0;
        size_t retval_size = sizeof(retval);
        ret = DkStreamWrite(DkGetPalPublicState()->parent_process, 0, &retval_size, &retval, NULL);
        if (ret < 0 || retval_size != sizeof(retval)) {
            pal_printf("DkStreamWrite failed\n");
            return 1;
        }
    }

    return 0;
}
