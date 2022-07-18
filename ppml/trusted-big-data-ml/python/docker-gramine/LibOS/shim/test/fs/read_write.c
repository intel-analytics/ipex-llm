#include "common.h"

static void read_write(const char* file_path) {
    const size_t size = 1024 * 1024;
    int fd = open_output_fd(file_path, /*rdwr=*/true);
    printf("open(%s) RW OK\n", file_path);

    void* buf1 = alloc_buffer(size);
    void* buf2 = alloc_buffer(size);
    fill_random(buf1, size);
    write_fd(file_path, fd, buf1, size);
    printf("write(%s) RW OK\n", file_path);
    seek_fd(file_path, fd, 0, SEEK_SET);
    printf("seek(%s) RW OK\n", file_path);
    read_fd(file_path, fd, buf2, size);
    printf("read(%s) RW OK\n", file_path);
    if (memcmp(buf1, buf2, size) != 0)
        fatal_error("Read data is different from what was written\n");
    printf("compare(%s) RW OK\n", file_path);

    for (size_t i = 0; i < 1024; i++) {
        size_t offset = rand() % (size - 1024);
        size_t chunk_size = rand() % 1024;
        fill_random(buf1, chunk_size);
        seek_fd(file_path, fd, offset, SEEK_SET);
        write_fd(file_path, fd, buf1, chunk_size);
        seek_fd(file_path, fd, offset, SEEK_SET);
        read_fd(file_path, fd, buf2, chunk_size);
        if (memcmp(buf1, buf2, chunk_size) != 0)
            fatal_error("Chunk data is different from what was written (offset %zu, size %zu)\n",
                        offset, chunk_size);
    }

    close_fd(file_path, fd);
    printf("close(%s) RW OK\n", file_path);
    free(buf1);
    free(buf2);
}

int main(int argc, char* argv[]) {
    if (argc < 2)
        fatal_error("Usage: %s <file_path>\n", argv[0]);

    setup();
    read_write(argv[1]);
    return 0;
}
