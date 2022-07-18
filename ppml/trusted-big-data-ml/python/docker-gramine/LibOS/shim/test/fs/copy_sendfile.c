#include "common.h"

void copy_data(int fi, int fo, const char* input_path, const char* output_path, size_t size) {
    sendfile_fd(input_path, output_path, fi, fo, size);
    printf("sendfile_fd(%zu) OK\n", size);
}
