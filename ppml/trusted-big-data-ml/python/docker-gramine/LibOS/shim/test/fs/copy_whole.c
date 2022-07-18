#include "common.h"

void copy_data(int fi, int fo, const char* input_path, const char* output_path, size_t size) {
    void* data = alloc_buffer(size);
    read_fd(input_path, fi, data, size);
    printf("read_fd(%zu) input OK\n", size);
    write_fd(output_path, fo, data, size);
    printf("write_fd(%zu) output OK\n", size);
    free(data);
}
