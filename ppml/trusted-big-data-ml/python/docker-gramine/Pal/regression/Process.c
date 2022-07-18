#include "api.h"
#include "pal.h"
#include "pal_regression.h"

int main(int argc, char** argv, char** envp) {
    char buffer1[20] = "Hello World 1", buffer2[20] = "Hello World 2";
    char buffer3[20], buffer4[20];
    int ret;
    size_t size;

    if (argc > 1 && !memcmp(argv[1], "Child", 6)) {
        pal_printf("Child Process Created\n");

        /* check arguments */
        pal_printf("# of Arguments: %d\n", argc);
        for (int i = 0; i < argc; i++) {
            pal_printf("argv[%d] = %s\n", i, argv[i]);
        }

        size = sizeof(buffer1);
        DkStreamWrite(DkGetPalPublicState()->parent_process, 0, &size, buffer1, NULL);

        size = sizeof(buffer1);
        ret = DkStreamWrite(DkGetPalPublicState()->parent_process, 0, &size, buffer1, NULL);
        if (ret == 0 && size > 0)
            pal_printf("Process Write 1 OK\n");

        size = sizeof(buffer4);
        ret = DkStreamRead(DkGetPalPublicState()->parent_process, 0, &size, buffer4, NULL, 0);
        if (ret == 0 && size > 0)
            pal_printf("Process Read 2: %s\n", buffer4);

    } else {
        const char* args[3] = {"Process", "Child", 0};
        PAL_HANDLE children[3] = { 0 };

        for (int i = 0; i < 3; i++) {
            pal_printf("Creating process\n");

            ret = DkProcessCreate(args, &children[i]);

            if (ret == 0 && children[i]) {
                pal_printf("Process created %d\n", i + 1);
                size = sizeof(buffer4);
                DkStreamRead(children[i], 0, &size, buffer4, NULL, 0);
            }
        }

        for (int i = 0; i < 3; i++)
            if (children[i]) {
                size = sizeof(buffer3);
                ret = DkStreamRead(children[i], 0, &size, buffer3, NULL, 0);
                if (ret == 0 && size > 0)
                    pal_printf("Process Read 1: %s\n", buffer3);

                size = sizeof(buffer2);
                ret = DkStreamWrite(children[i], 0, &size, buffer2, NULL);
                if (ret == 0 && size > 0)
                    pal_printf("Process Write 2 OK\n");
            }
    }

    return 0;
}
