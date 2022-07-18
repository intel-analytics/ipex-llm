#include "common.h"

void copy_data(int fi, int fo, const char* input_path, const char* output_path, size_t size) {
    size_t max_step = 16;
    if (size > 65536)
        max_step = 256;

    void* data = alloc_buffer(max_step);
    size_t offset = 0;
    size_t step;
    while (offset < size) {
        if (offset + max_step <= size)
            step = rand() % max_step + 1;
        else
            step = size - offset;
        read_fd(input_path, fi, data, step);
        write_fd(output_path, fo, data, step);
        offset += step;
    }
    free(data);
}
