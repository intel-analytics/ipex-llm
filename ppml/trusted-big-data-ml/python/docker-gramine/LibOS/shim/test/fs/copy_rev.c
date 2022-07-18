#include "common.h"

void copy_data(int fi, int fo, const char* input_path, const char* output_path, size_t size) {
    assert(!OVERFLOWS(off_t, size));

    size_t max_step = 16;
    if (size > 65536)
        max_step = 256;

    void* data = alloc_buffer(max_step);
    off_t offset = size;
    size_t step;
    while (offset > 0) {
        if (offset > (off_t)max_step)
            step = rand() % max_step + 1;
        else
            step = offset;
        offset -= step;
        seek_fd(input_path, fi, offset, SEEK_SET);
        seek_fd(output_path, fo, offset, SEEK_SET);
        read_fd(input_path, fi, data, step);
        write_fd(output_path, fo, data, step);
    }
    free(data);
}
