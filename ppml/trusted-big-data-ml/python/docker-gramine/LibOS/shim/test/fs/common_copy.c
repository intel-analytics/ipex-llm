#include "common.h"

#ifdef COPY_MMAP
#define RDWR_OUTPUT_OPEN true
#else
#define RDWR_OUTPUT_OPEN false
#endif

static void copy_file(const char* input_path, const char* output_path, size_t size) {
    assert(!OVERFLOWS(off_t, size));

    int fi = open_input_fd(input_path);
    printf("open(%zu) input OK\n", size);

    struct stat st;
    if (fstat(fi, &st) < 0)
        fatal_error("Failed to stat file %s: %s\n", input_path, strerror(errno));
    if (st.st_size != (off_t)size)
        fatal_error("Size mismatch: expected %zu, got %zu\n", size, st.st_size);
    printf("fstat(%zu) input OK\n", size);

    int fo = open_output_fd(output_path, /*rdwr=*/RDWR_OUTPUT_OPEN);
    printf("open(%zu) output OK\n", size);

    if (fstat(fo, &st) < 0)
        fatal_error("Failed to stat file %s: %s\n", output_path, strerror(errno));
    if (st.st_size != 0)
        fatal_error("Size mismatch: expected 0, got %zu\n", st.st_size);
    printf("fstat(%zu) output 1 OK\n", size);

    copy_data(fi, fo, input_path, output_path, size);

    if (fstat(fo, &st) < 0)
        fatal_error("Failed to stat file %s: %s\n", output_path, strerror(errno));
    if (st.st_size != (off_t)size)
        fatal_error("Size mismatch: expected %zu, got %zu\n", size, st.st_size);
    printf("fstat(%zu) output 2 OK\n", size);

    close_fd(input_path, fi);
    printf("close(%zu) input OK\n", size);
    close_fd(output_path, fo);
    printf("close(%zu) output OK\n", size);
}

int main(int argc, char* argv[]) {
    if (argc < 3)
        fatal_error("Usage: %s <input_dir> <output_dir>\n", argv[0]);

    setup();

    char* input_dir = argv[1];
    char* output_dir = argv[2];

    // Process input directory
    DIR* dfd = opendir(input_dir);
    if (!dfd)
        fatal_error("Failed to open input directory %s: %s\n", input_dir, strerror(errno));
    printf("opendir(%s) OK\n", input_dir);

    struct dirent* de = NULL;
    while ((de = readdir(dfd)) != NULL) {
        printf("readdir(%s) OK\n", de->d_name);
        if (!strcmp(de->d_name, "."))
            continue;
        if (!strcmp(de->d_name, ".."))
            continue;

        // assume files have names that are their sizes as string
        size_t input_path_size = strlen(input_dir) + 1 + strlen(de->d_name) + 1;
        size_t output_path_size = strlen(output_dir) + 1 + strlen(de->d_name) + 1;
        char* input_path = alloc_buffer(input_path_size);
        char* output_path = alloc_buffer(output_path_size);
        size_t size = (size_t)strtoumax(de->d_name, NULL, 10);
        snprintf(input_path, input_path_size, "%s/%s", input_dir, de->d_name);
        snprintf(output_path, output_path_size, "%s/%s", output_dir, de->d_name);

        copy_file(input_path, output_path, size);

        free(input_path);
        free(output_path);
    }

    return 0;
}
