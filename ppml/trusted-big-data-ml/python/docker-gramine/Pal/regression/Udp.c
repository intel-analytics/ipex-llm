#include "api.h"
#include "pal.h"
#include "pal_regression.h"

#define NTRIES 10

char addr[40];
char buffer[20];

int main(int argc, char** argv) {
    int i;
    int ret;

    if (argc == 1) {
        unsigned long start = 0;
        if (DkSystemTimeQuery(&start) < 0) {
            pal_printf("DkSystemTimeQuery failed\n");
            return 1;
        }

        const char* newargs[3] = {"Udp", "child", NULL};

        PAL_HANDLE srv = NULL;
        ret = DkStreamOpen("udp.srv:127.0.0.1:8000", PAL_ACCESS_RDWR, /*share_flags=*/0,
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
            pal_printf("Udp: DkProcessCreate failed\n");
            return 1;
        }

        for (i = 0; i < NTRIES; i++) {
            size_t bytes = sizeof(buffer);
            ret = DkStreamRead(srv, 0, &bytes, buffer, addr, sizeof(addr));

            if (ret < 0 || bytes == 0) {
                pal_printf("not able to receive from client\n");
                return 1;
            }

            pal_printf("read on server (from %s): %s\n", addr, buffer);
        }

        unsigned long end = 0;
        if (DkSystemTimeQuery(&end) < 0) {
            pal_printf("DkSystemTimeQuery failed\n");
            return 1;
        }
        pal_printf("wall time = %ld\n", end - start);

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
        PAL_HANDLE cli = NULL;
        ret = DkStreamOpen("udp:127.0.0.1:8000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                           PAL_CREATE_IGNORED, /*options=*/0, &cli);
        if (ret < 0) {
            pal_printf("DkStreamOpen failed\n");
            return 1;
        }

        ret = DkStreamGetName(cli, addr, sizeof(addr));
        if (ret < 0) {
            pal_printf("DkStreamGetName failed\n");
            return 1;
        }
        pal_printf("client connected on %s\n", addr);

        for (i = 0; i < NTRIES; i++) {
            char buf[12] = "Hello World";
            size_t buf_size = sizeof(buf);
            ret = DkStreamWrite(cli, 0, &buf_size, buf, NULL);

            if (ret < 0 || buf_size != sizeof(buf)) {
                pal_printf("not able to send to server\n");
                return 1;
            }
        }

        DkObjectClose(cli);

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
