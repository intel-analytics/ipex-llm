#include "api.h"
#include "pal.h"
#include "pal_regression.h"

#define NUM_TO_HEX(num) ((num) >= 10 ? 'a' + ((num) - 10) : '0' + (num))
#define BUF_SIZE        40

char buffer1[BUF_SIZE];
char buffer2[BUF_SIZE];
char hex_buf[BUF_SIZE * 2 + 1];

static void print_hex(const char* fmt, const void* data, size_t size) {
    hex_buf[size * 2] = '\0';
    for (size_t i = 0; i < size; i++) {
        unsigned char b = ((unsigned char*)data)[i];
        hex_buf[i * 2]     = NUM_TO_HEX(b >> 4);
        hex_buf[i * 2 + 1] = NUM_TO_HEX(b & 0xf);
    }
    pal_printf(fmt, hex_buf);
}

int main(int argc, char** argv, char** envp) {
    int ret;

    /* test regular file opening */

    PAL_HANDLE file1 = NULL;
    ret = DkStreamOpen("file:File.manifest", PAL_ACCESS_RDWR, /*share_flags=*/0,
                       PAL_CREATE_NEVER, /*options=*/0, &file1);
    if (ret >= 0 && file1) {
        pal_printf("File Open Test 1 OK\n");

        /* test file read */
        size_t size = sizeof(buffer1);
        ret = DkStreamRead(file1, 0, &size, buffer1, NULL, 0);
        if (ret == 0 && size == sizeof(buffer1)) {
            print_hex("Read Test 1 (0th - 40th): %s\n", buffer1, size);
        }

        size = sizeof(buffer1);
        ret = DkStreamRead(file1, 0, &size, buffer1, NULL, 0);
        if (ret == 0 && size == sizeof(buffer1)) {
            print_hex("Read Test 2 (0th - 40th): %s\n", buffer1, size);
        }

        size = sizeof(buffer2);
        ret = DkStreamRead(file1, 200, &size, buffer2, NULL, 0);
        if (ret == 0 && size == sizeof(buffer2)) {
            print_hex("Read Test 3 (200th - 240th): %s\n", buffer2, size);
        }

        /* test file attribute query */

        PAL_STREAM_ATTR attr1;
        ret = DkStreamAttributesQueryByHandle(file1, &attr1);
        if (ret >= 0) {
            pal_printf("Query by Handle: type = %d, size = %ld\n", attr1.handle_type,
                       attr1.pending_size);
        }

        /* test file map */

        void* mem1 = (void*)DkGetPalPublicState()->user_address_start;
        ret = DkStreamMap(file1, &mem1, PAL_PROT_READ | PAL_PROT_WRITECOPY, 0, PAGE_SIZE);
        if (ret >= 0 && mem1) {
            memcpy(buffer1, mem1, 40);
            print_hex("Map Test 1 (0th - 40th): %s\n", buffer1, 40);

            memcpy(buffer2, mem1 + 200, 40);
            print_hex("Map Test 2 (200th - 240th): %s\n", buffer2, 40);

            ret = DkStreamUnmap(mem1, PAGE_SIZE);
            if (ret < 0) {
                pal_printf("DkStreamUnmap failed\n");
                return 1;
            }
        } else {
            pal_printf("Map Test 1 & 2: Failed to map buffer\n");
        }

        DkObjectClose(file1);
    }

    PAL_HANDLE file2 = NULL;
    ret = DkStreamOpen("file:File.manifest", PAL_ACCESS_RDWR, /*share_flags=*/0,
                       PAL_CREATE_NEVER, /*options=*/0, &file2);
    if (ret >= 0 && file2) {
        pal_printf("File Open Test 2 OK\n");
        DkObjectClose(file2);
    }

    PAL_HANDLE file3 = NULL;
    ret = DkStreamOpen("file:../regression/File.manifest", PAL_ACCESS_RDWR, /*share_flags=*/0,
                       PAL_CREATE_NEVER, /*options=*/0, &file3);
    if (ret >= 0 && file3) {
        pal_printf("File Open Test 3 OK\n");
        DkObjectClose(file3);
    }

    PAL_STREAM_ATTR attr2;
    ret = DkStreamAttributesQuery("file:File.manifest", &attr2);
    if (ret >= 0) {
        pal_printf("Query: type = %d, size = %ld\n", attr2.handle_type, attr2.pending_size);
    }

    /* test regular file creation */

    PAL_HANDLE file4 = NULL;
    ret = DkStreamOpen("file:file_nonexist.tmp", PAL_ACCESS_RDWR,
                       PAL_SHARE_OWNER_R | PAL_SHARE_OWNER_W, PAL_CREATE_ALWAYS, /*options=*/0,
                       &file4);
    if (ret >= 0 && file4)
        pal_printf("File Creation Test 1 OK\n");

    PAL_HANDLE file5 = NULL;
    ret = DkStreamOpen("file:file_nonexist.tmp", PAL_ACCESS_RDWR,
                       PAL_SHARE_OWNER_R | PAL_SHARE_OWNER_W, PAL_CREATE_ALWAYS, /*options=*/0,
                       &file5);
    if (ret >= 0) {
        DkObjectClose(file5);
    } else {
        pal_printf("File Creation Test 2 OK\n");
    }

    PAL_HANDLE file6 = NULL;
    ret = DkStreamOpen("file:file_nonexist.tmp", PAL_ACCESS_RDWR,
                       PAL_SHARE_OWNER_R | PAL_SHARE_OWNER_W, PAL_CREATE_TRY, /*options=*/0,
                       &file6);
    if (ret >= 0 && file6) {
        pal_printf("File Creation Test 3 OK\n");
        DkObjectClose(file6);
    }

    if (file4) {
        /* test file writing */

        size_t size = sizeof(buffer1);
        ret = DkStreamWrite(file4, 0, &size, buffer1, NULL);
        if (ret < 0)
            goto fail_writing;

        size = sizeof(buffer2);
        ret = DkStreamWrite(file4, 0, &size, buffer2, NULL);
        if (ret < 0)
            goto fail_writing;

        size = sizeof(buffer1);
        ret = DkStreamWrite(file4, 200, &size, buffer1, NULL);
        if (ret < 0)
            goto fail_writing;

        /* test file truncate */
        ret = DkStreamSetLength(file4, DkGetPalPublicState()->alloc_align);
        if (ret < 0) {
            goto fail_writing;
        }

    fail_writing:
        DkObjectClose(file4);
        if (ret < 0) {
            return 1;
        }
    }

    PAL_HANDLE file7 = NULL;
    ret = DkStreamOpen("file:file_delete.tmp", PAL_ACCESS_RDONLY, /*share_flags=*/0,
                       PAL_CREATE_NEVER, /*options=*/0, &file7);
    if (ret >= 0 && file7) {
        ret = DkStreamDelete(file7, PAL_DELETE_ALL);
        if (ret < 0) {
            pal_printf("DkStreamDelete failed\n");
            return 1;
        }
        DkObjectClose(file7);
    }

    return 0;
}
