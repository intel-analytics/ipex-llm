#include "common.h"

static void file_delete(const char* file_path_1, const char* file_path_2, bool writable) {
    const char* type = writable ? "output" : "input";

    int fd = writable ? open_output_fd(file_path_1, /*rdwr=*/false) : open_input_fd(file_path_1);
    printf("open(%s) %s 1 OK\n", file_path_1, type);

    close_fd(file_path_1, fd);
    printf("close(%s) %s 1 OK\n", file_path_1, type);

    if (unlink(file_path_1) != 0)
        fatal_error("Failed to unlink file %s: %s\n", file_path_1, strerror(errno));
    printf("unlink(%s) %s 1 OK\n", file_path_1, type);

    fd = writable ? open_output_fd(file_path_2, /*rdwr=*/false) : open_input_fd(file_path_2);
    printf("open(%s) %s 2 OK\n", file_path_2, type);

    if (unlink(file_path_2) != 0)
        fatal_error("Failed to unlink file %s: %s\n", file_path_2, strerror(errno));
    printf("unlink(%s) %s 2 OK\n", file_path_2, type);

    close_fd(file_path_2, fd);
    printf("close(%s) %s 2 OK\n", file_path_2, type);
}

int main(int argc, char* argv[]) {
    if (argc < 6)
        fatal_error("Usage: %s <path1> <path2> <path3> <path4> <path5>\n", argv[0]);

    setup();
    if (unlink(argv[1]) != 0)
        fatal_error("Failed to unlink file %s: %s\n", argv[1], strerror(errno));
    printf("unlink(%s) OK\n", argv[1]);

    file_delete(argv[2], argv[3], /*writable=*/false);
    file_delete(argv[4], argv[5], /*writable=*/true);

    return 0;
}
