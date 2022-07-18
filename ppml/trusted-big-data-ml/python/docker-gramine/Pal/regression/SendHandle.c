#include "api.h"
#include "pal.h"
#include "pal_regression.h"

int main(int argc, char** argv) {
    int ret;
    size_t size;
    PAL_HANDLE handles[3];

    if (argc == 2 && !memcmp(argv[1], "Child", 6)) {
        char buffer[20];

        for (int i = 0; i < 3; i++) {
            handles[i] = NULL;
            ret = DkReceiveHandle(DkGetPalPublicState()->parent_process, &handles[i]);
            if (ret >= 0 && handles[i]) {
                pal_printf("Receive Handle OK\n");
            } else {
                continue;
            }

            memset(buffer, 0, 20);

            switch (i) {
                case 0:; /* pipe.srv */
                    PAL_HANDLE pipe = NULL;
                    ret = DkStreamWaitForClient(handles[i], &pipe, /*options=*/0);

                    if (ret >= 0 && pipe) {
                        size = sizeof(buffer);
                        ret = DkStreamRead(pipe, 0, &size, buffer, NULL, 0);
                        if (ret == 0 && size > 0)
                            pal_printf("Receive Pipe Handle: %s\n", buffer);

                        DkObjectClose(pipe);
                    }

                    break;

                case 1:; /* udp.srv */
                    char uri[20];

                    size = sizeof(buffer);
                    ret = DkStreamRead(handles[i], 0, &size, buffer, uri, sizeof(uri));
                    if (ret == 0 && size > 0)
                        pal_printf("Receive Socket Handle: %s\n", buffer);

                    break;

                case 2: /* file */
                    size = sizeof(buffer);
                    ret = DkStreamRead(handles[i], 0, &size, buffer, NULL, 0);
                    if (ret == 0 && size > 0)
                        pal_printf("Receive File Handle: %s\n", buffer);

                    break;

                default:
                    break;
            }

            DkObjectClose(handles[i]);
        }
    } else {
        const char* args[3] = {"SendHandle", "Child", NULL};

        PAL_HANDLE child = NULL;
        ret = DkProcessCreate(args, &child);

        if (ret >= 0 && child) {
            // Sending pipe handle
            ret = DkStreamOpen("pipe.srv:1", PAL_ACCESS_RDWR, 0, PAL_CREATE_IGNORED, 0,
                               &handles[0]);

            if (ret >= 0 && handles[0]) {
                pal_printf("Send Handle OK\n");

                if (DkSendHandle(child, handles[0]) >= 0) {
                    DkObjectClose(handles[0]);
                    PAL_HANDLE pipe = NULL;
                    ret = DkStreamOpen("pipe:1", PAL_ACCESS_RDWR, /*share_flags=*/0,
                                       PAL_CREATE_IGNORED, /*options=*/0, &pipe);
                    if (ret >= 0 && pipe) {
                        char buf[20] = "Hello World";
                        size_t buf_size = sizeof(buf);
                        DkStreamWrite(pipe, 0, &buf_size, buf, NULL);
                        DkObjectClose(pipe);
                    }
                } else {
                    DkObjectClose(handles[0]);
                }
            }

            // Sending udp handle
            ret = DkStreamOpen("udp.srv:127.0.0.1:8000", PAL_ACCESS_RDWR, 0, PAL_CREATE_IGNORED, 0,
                               &handles[1]);

            if (ret >= 0 && handles[1]) {
                pal_printf("Send Handle OK\n");

                if (DkSendHandle(child, handles[1]) >= 0) {
                    DkObjectClose(handles[1]);
                    PAL_HANDLE socket = NULL;
                    ret = DkStreamOpen("udp:127.0.0.1:8000", PAL_ACCESS_RDWR, /*share_flags=*/0,
                                       PAL_CREATE_IGNORED, /*options=*/0, &socket);
                    if (ret >= 0 && socket) {
                        char buf[20] = "Hello World";
                        size_t buf_size = sizeof(buf);
                        DkStreamWrite(socket, 0, &buf_size, buf, NULL);
                        DkObjectClose(socket);
                    }
                } else {
                    DkObjectClose(handles[1]);
                }
            }

            ret = DkStreamOpen("file:to_send.tmp", PAL_ACCESS_RDWR, 0600, PAL_CREATE_TRY, 0,
                               &handles[2]);

            if (ret >= 0 && handles[2]) {
                pal_printf("Send Handle OK\n");

                char buf[20] = "Hello World";
                size_t buf_size = sizeof(buf);
                DkStreamWrite(handles[2], 0, &buf_size, buf, NULL);
                DkStreamSetLength(handles[2], 4096);

                DkSendHandle(child, handles[2]);
                DkObjectClose(handles[2]);
            }
        }

        DkObjectClose(child);
    }

    return 0;
}
