#include "common.h"

const int g_mode = 0664;
const char g_data = 'x';

static size_t get_file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) < 0)
        fatal_error("Failed to stat file '%s': %s\n", path, strerror(errno));
    printf("size(%s) == %zu\n", path, st.st_size);
    return st.st_size;
}

static void open_test__(const char* path, int flags, const char* flags_str, bool exists,
                        bool expect_success, bool do_write) {
    const char* exists_str = exists ? "exists" : "doesn't exist";
    int fd = open(path, flags, g_mode);
    if (fd < 0) {
        if (expect_success) {
            fatal_error("open(%s) [%s] failed!\n", flags_str, exists_str);
        } else {
            printf("open(%s) [%s] failed as expected\n", flags_str, exists_str);
        }
    } else {
        if (expect_success) {
            printf("open(%s) [%s] succeeded as expected\n", flags_str, exists_str);
            if (do_write)
                write_fd(path, fd, &g_data, sizeof(g_data));
        } else {
            fatal_error("open(%s) [%s] unexpectedly succeeded\n", flags_str, exists_str);
        }
        close(fd);
    }
}

#define OPEN_TEST(path, flags, exists, expect_success, do_write) \
    open_test__(path, flags, #flags, exists, expect_success, do_write)

int main(int argc, char* argv[]) {
    if (argc < 2)
        fatal_error("Usage: %s <path>\n", argv[0]);

    setup();

    // doesn't exist - should create
    OPEN_TEST(argv[1], O_CREAT | O_EXCL | O_RDWR, /*exists=*/false, /*expect_success=*/true,
              /*do_write=*/true);

    // exists - open should fail
    OPEN_TEST(argv[1], O_CREAT | O_EXCL | O_RDWR, /*exists=*/true, /*expect_success=*/false,
              /*do_write=*/false);

    // exists - should open existing and NOT truncate
    OPEN_TEST(argv[1], O_CREAT | O_RDWR, /*exists=*/true, /*expect_success=*/true,
              /*do_write=*/false);
    if (get_file_size(argv[1]) != 1)
        fatal_error("File was truncated\n");

    if (unlink(argv[1]) < 0)
        fatal_error("unlink(%s) failed: %s\n", argv[1], strerror(errno));

    // doesn't exist - should create new
    OPEN_TEST(argv[1], O_CREAT | O_RDWR, /*exists=*/false, /*expect_success=*/true,
              /*do_write=*/false);

    if (unlink(argv[1]) < 0)
        fatal_error("unlink(%s) failed: %s\n", argv[1], strerror(errno));

    // doesn't exist - should create new
    OPEN_TEST(argv[1], O_CREAT | O_TRUNC | O_RDWR, /*exists=*/false, /*expect_success=*/true,
              /*do_write=*/true);

    // exists - should truncate
    OPEN_TEST(argv[1], O_CREAT | O_TRUNC | O_RDWR, /*exists=*/true, /*expect_success=*/true,
              /*do_write=*/false);
    if (get_file_size(argv[1]) != 0)
        fatal_error("File was not truncated\n");

    if (unlink(argv[1]) < 0)
        fatal_error("unlink(%s) failed: %s\n", argv[1], strerror(errno));

    return 0;
}
