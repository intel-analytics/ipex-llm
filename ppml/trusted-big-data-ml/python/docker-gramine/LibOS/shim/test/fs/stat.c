#include "common.h"

static void file_stat(const char* file_path, bool writable) {
    struct stat st;
    const char* type = writable ? "output" : "input";

    if (stat(file_path, &st) != 0)
        fatal_error("Failed to stat file %s: %s\n", file_path, strerror(errno));
    printf("stat(%s) %s 1 OK: %zu\n", file_path, type, st.st_size);

    int fd = writable ? open_output_fd(file_path, /*rdwr=*/false) : open_input_fd(file_path);
    printf("open(%s) %s 2 OK\n", file_path, type);

    if (stat(file_path, &st) != 0)
        fatal_error("Failed to stat file %s: %s\n", file_path, strerror(errno));
    printf("stat(%s) %s 2 OK: %zu\n", file_path, type, st.st_size);

    if (fstat(fd, &st) != 0)
        fatal_error("Failed to fstat file %s: %s\n", file_path, strerror(errno));
    printf("fstat(%s) %s 2 OK: %zu\n", file_path, type, st.st_size);

    close_fd(file_path, fd);
    printf("close(%s) %s 2 OK\n", file_path, type);
}

int main(int argc, char* argv[]) {
    if (argc < 3)
        fatal_error("Usage: %s <input_path> <output_path>\n", argv[0]);

    setup();
    file_stat(argv[1], false);
    file_stat(argv[2], true);

    return 0;
}
