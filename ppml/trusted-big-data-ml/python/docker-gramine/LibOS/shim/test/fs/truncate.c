#include "common.h"

static void file_truncate(const char* file_path_1, const char* file_path_2, size_t size) {
    if (truncate(file_path_1, size) != 0)
        fatal_error("Failed to truncate file %s to %zu: %s\n", file_path_1, size, strerror(errno));
    printf("truncate(%s) to %zu OK\n", file_path_1, size);

    int fd = open_output_fd(file_path_2, /*rdwr=*/false);
    printf("open(%s) output OK\n", file_path_2);

    if (ftruncate(fd, size) != 0)
        fatal_error("Failed to ftruncate file %s to %zu: %s\n", file_path_2, size, strerror(errno));
    printf("ftruncate(%s) to %zu OK\n", file_path_2, size);

    close_fd(file_path_2, fd);
    printf("close(%s) output OK\n", file_path_2);
}

int main(int argc, char* argv[]) {
    if (argc < 4)
        fatal_error("Usage: %s <file_path_1> <file_path_2> <size>\n", argv[0]);

    setup();
    size_t size = strtoul(argv[3], NULL, 10);
    file_truncate(argv[1], argv[2], size);

    return 0;
}
