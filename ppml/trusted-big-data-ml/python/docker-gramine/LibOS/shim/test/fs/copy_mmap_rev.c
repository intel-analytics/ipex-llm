#include "common.h"

void copy_data(int fi, int fo, const char* input_path, const char* output_path, size_t size) {
    if (size > 0) {
        size_t max_step = 16;
        if (size > 65536)
            max_step = 256;

        // map whole input/output file
        void* in = mmap_fd(input_path, fi, PROT_READ, 0, size);
        printf("mmap_fd(%zu) input OK\n", size);
        void* out = mmap_fd(output_path, fo, PROT_WRITE, 0, size);
        printf("mmap_fd(%zu) output OK\n", size);

        // copy data
        if (ftruncate(fo, size) != 0)
            fatal_error("ftruncate(%s, %zu) failed: %s\n", output_path, size, strerror(errno));
        printf("ftruncate(%zu) output OK\n", size);

        size_t offset = size;
        size_t step;
        while (offset > 0) {
            if (offset > max_step)
                step = rand() % max_step + 1;
            else
                step = offset;
            offset -= step;

            memcpy(out + offset, in + offset, step);
        }

        // unmap
        munmap_fd(input_path, in, size);
        printf("munmap_fd(%zu) input OK\n", size);
        munmap_fd(output_path, out, size);
        printf("munmap_fd(%zu) output OK\n", size);
    }
}
