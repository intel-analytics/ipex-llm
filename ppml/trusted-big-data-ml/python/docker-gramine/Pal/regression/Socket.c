#include "api.h"
#include "pal.h"
#include "pal_regression.h"

/* Checks if stream matches `pattern`, and aborts otherwise. `*` in the pattern is interpreted as
 * one or more digits, and we use it to match the port number assigned by system. */
static void expect_name(PAL_HANDLE handle, const char* pattern) {
    char name[128];
    int ret = DkStreamGetName(handle, name, sizeof(name));
    if (ret < 0) {
        pal_printf("DkStreamGetName failed: %d\n", ret);
        DkProcessExit(1);
    }

    const char* pn = name;
    const char* pp = pattern;
    while (*pn && *pp) {
        if (*pp == '*') {
            /* consume one or more digits */
            if (!isdigit(*pn))
                break;
            while (isdigit(*pn))
                pn++;
            pp++;
        } else if (*pn == *pp) {
            /* identical characters in name and pattern */
            pn++;
            pp++;
        } else {
            /* mismatch */
            break;
        }
    }
    if (*pn || *pp) {
        pal_printf("Wrong stream name: %s, expected: %s\n", name, pattern);
        DkProcessExit(1);
    }
}

int main(int argc, char** argv, char** envp) {
    char buffer1[20] = "Hello World 1", buffer2[20] = "Hello World 2";
    char buffer3[20], buffer4[20];
    int ret;
    size_t size;

    memset(buffer3, 0, 20);
    memset(buffer4, 0, 20);

    PAL_HANDLE tcp1 = NULL;
    ret = DkStreamOpen("tcp.srv:127.0.0.1:3000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                       PAL_CREATE_IGNORED, /*options=*/0, &tcp1);

    if (ret >= 0 && tcp1) {
        pal_printf("TCP Creation 1 OK\n");
        expect_name(tcp1, "tcp.srv:127.0.0.1:3000");

        PAL_HANDLE tcp2 = NULL;
        ret = DkStreamOpen("tcp:127.0.0.1:3000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                           PAL_CREATE_IGNORED, /*options=*/0, &tcp2);

        if (ret >= 0 && tcp2) {
            /* TODO: Linux-SGX reports just "127.0.0.1:3000" here (i.e. omits the source address) */
            // expect_name(tcp2, "tcp:127.0.0.1:*:127.0.0.1:3000");

            PAL_HANDLE tcp3 = NULL;
            ret = DkStreamWaitForClient(tcp1, &tcp3, /*options=*/0);

            if (ret >= 0 && tcp3) {
                pal_printf("TCP Connection 1 OK\n");
                expect_name(tcp3, "tcp:127.0.0.1:3000:127.0.0.1:*");

                size = sizeof(buffer1);
                ret = DkStreamWrite(tcp3, 0, &size, buffer1, NULL);
                if (ret == 0 && size > 0)
                    pal_printf("TCP Write 1 OK\n");

                size = sizeof(buffer3);
                ret = DkStreamRead(tcp2, 0, &size, buffer3, NULL, 0);
                if (ret == 0 && size > 0)
                    pal_printf("TCP Read 1: %s\n", buffer3);

                size = sizeof(buffer2);
                ret = DkStreamWrite(tcp2, 0, &size, buffer2, NULL);
                if (ret == 0 && size > 0)
                    pal_printf("TCP Write 2 OK\n");

                size = sizeof(buffer4);
                ret = DkStreamRead(tcp3, 0, &size, buffer4, NULL, 0);
                if (ret == 0 && size > 0)
                    pal_printf("TCP Read 2: %s\n", buffer4);

                DkObjectClose(tcp3);
            }

            DkObjectClose(tcp2);
        }

        ret = DkStreamDelete(tcp1, PAL_DELETE_ALL);
        if (ret < 0) {
            pal_printf("DkStreamDelete failed\n");
            return 1;
        }
        DkObjectClose(tcp1);
    }

    /* Test IPv6 (actually, IPv4 connection to an IPv6 socket; a true IPv6 connection will not work
     * inside a Docker host without extra configuration) */

    PAL_HANDLE tcp1_ipv6 = NULL;
    ret = DkStreamOpen("tcp.srv:[::]:3000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                       PAL_CREATE_IGNORED, PAL_OPTION_DUALSTACK, &tcp1_ipv6);
    if (ret >= 0 && tcp1_ipv6) {
        pal_printf("TCP (IPv6) Creation 1 OK\n");
        expect_name(tcp1_ipv6, "tcp.srv:[0000:0000:0000:0000:0000:0000:0000:0000]:3000");

        PAL_HANDLE tcp2_ipv6 = NULL;
        ret = DkStreamOpen("tcp:127.0.0.1:3000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                           PAL_CREATE_IGNORED, /*options=*/0, &tcp2_ipv6);
        if (ret >= 0 && tcp2_ipv6) {
            /* TODO: Linux-SGX reports just "127.0.0.1:3000" here (i.e. omits the source address) */
            // expect_name(tcp2_ipv6, "tcp:127.0.0.1:*:127.0.0.1:3000");

            PAL_HANDLE tcp3_ipv6 = NULL;
            ret = DkStreamWaitForClient(tcp1_ipv6, &tcp3_ipv6, /*options=*/0);

            if (ret >= 0 && tcp3_ipv6) {
                pal_printf("TCP (IPv6) Connection 1 OK\n");
                expect_name(tcp3_ipv6,
                            "tcp:[0000:0000:0000:0000:0000:0000:0000:0000]:3000:"
                            "[0000:0000:0000:0000:0000:ffff:7f00:0001]:*");

                DkObjectClose(tcp3_ipv6);
            }

            DkObjectClose(tcp2_ipv6);
        }

        ret = DkStreamDelete(tcp1_ipv6, PAL_DELETE_ALL);
        if (ret < 0) {
            pal_printf("DkStreamDelete failed\n");
            return 1;
        }
        DkObjectClose(tcp1_ipv6);
    }

    PAL_HANDLE udp1 = NULL;
    ret = DkStreamOpen("udp.srv:127.0.0.1:3000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                       PAL_CREATE_IGNORED, /*options=*/0, &udp1);

    if (ret >= 0 && udp1) {
        pal_printf("UDP Creation 1 OK\n");
        expect_name(udp1, "udp.srv:127.0.0.1:3000");

        PAL_HANDLE udp2 = NULL;
        ret = DkStreamOpen("udp:127.0.0.1:3000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                           PAL_CREATE_IGNORED, /*options=*/0, &udp2);

        if (ret >= 0 && udp2) {
            expect_name(udp2, "udp:127.0.0.1:3000");
            pal_printf("UDP Connection 1 OK\n");

            memset(buffer3, 0, 20);
            memset(buffer4, 0, 20);

            size = sizeof(buffer1);
            ret = DkStreamWrite(udp2, 0, &size, buffer1, NULL);
            if (ret == 0 && size > 0)
                pal_printf("UDP Write 1 OK\n");

            char uri[20];

            size = sizeof(buffer3);
            ret = DkStreamRead(udp1, 0, &size, buffer3, uri, sizeof(uri));
            if (ret == 0 && size > 0)
                pal_printf("UDP Read 1: %s\n", buffer3);

            size = sizeof(buffer2);
            ret = DkStreamWrite(udp1, 0, &size, buffer2, uri);
            if (ret == 0 && size > 0)
                pal_printf("UDP Write 2 OK\n");

            size = sizeof(buffer4);
            ret = DkStreamRead(udp2, 0, &size, buffer4, NULL, 0);
            if (ret == 0 && size > 0)
                pal_printf("UDP Read 2: %s\n", buffer4);

            DkObjectClose(udp2);
        }

        PAL_HANDLE udp3 = NULL;
        ret = DkStreamOpen("udp:127.0.0.1:3001:127.0.0.1:3000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                           PAL_CREATE_IGNORED, /*options=*/0, &udp3);

        if (ret >= 0 && udp3) {
            expect_name(udp3, "udp:127.0.0.1:3001:127.0.0.1:3000");
            pal_printf("UDP Connection 2 OK\n");

            memset(buffer3, 0, 20);
            memset(buffer4, 0, 20);

            size = sizeof(buffer1);
            ret = DkStreamWrite(udp3, 0, &size, buffer1, NULL);
            if (ret == 0 && size > 0)
                pal_printf("UDP Write 3 OK\n");

            char uri[20];

            size = sizeof(buffer3);
            ret = DkStreamRead(udp1, 0, &size, buffer3, uri, sizeof(uri));
            if (ret == 0 && size > 0)
                pal_printf("UDP Read 3: %s\n", buffer3);

            size = sizeof(buffer2);
            ret = DkStreamWrite(udp1, 0, &size, buffer2, "udp:127.0.0.1:3001");
            if (ret == 0 && size > 0)
                pal_printf("UDP Write 4 OK\n");

            size = sizeof(buffer4);
            ret = DkStreamRead(udp3, 0, &size, buffer4, NULL, 0);
            if (ret == 0 && size > 0)
                pal_printf("UDP Read 4: %s\n", buffer4);

            DkObjectClose(udp3);
        }

        ret = DkStreamDelete(udp1, PAL_DELETE_ALL);
        if (ret < 0) {
            pal_printf("DkStreamDelete failed\n");
            return 1;
        }
        DkObjectClose(udp1);
    }

    return 0;
}
