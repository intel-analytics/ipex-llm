#include "common.h"

void copy_data(int fi, int fo, const char* input_path, const char* output_path, size_t size) {
    if (size > 0) {
        // map whole input/output file
        void* in = mmap_fd(input_path, fi, PROT_READ, 0, size);
        printf("mmap_fd(%zu) input OK\n", size);
        void* out = mmap_fd(output_path, fo, PROT_WRITE, 0, size);
        printf("mmap_fd(%zu) output OK\n", size);
        // copy data
        if (ftruncate(fo, size) != 0)
            fatal_error("ftruncate(%s, %zu) failed: %s\n", output_path, size, strerror(errno));
        printf("ftruncate(%zu) output OK\n", size);
        memcpy(out, in, size);
        // unmap
        munmap_fd(input_path, in, size);
        printf("munmap_fd(%zu) input OK\n", size);
        munmap_fd(output_path, out, size);
        printf("munmap_fd(%zu) output OK\n", size);
    }
}
