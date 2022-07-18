#include <ctype.h>

#include "common.h"

static void open_close_input_fd(const char* input_path) {
    int fi = open_input_fd(input_path);
    printf("open(%s) input OK\n", input_path);
    close_fd(input_path, fi);
    printf("close(%s) input OK\n", input_path);

    int f1 = open_input_fd(input_path);
    printf("open(%s) input 1 OK\n", input_path);
    int f2 = open_input_fd(input_path);
    printf("open(%s) input 2 OK\n", input_path);
    close_fd(input_path, f1);
    printf("close(%s) input 1 OK\n", input_path);
    close_fd(input_path, f2);
    printf("close(%s) input 2 OK\n", input_path);
}

static void open_close_input_stdio(const char* input_path) {
    FILE* fi = open_input_stdio(input_path);
    printf("fopen(%s) input OK\n", input_path);
    close_stdio(input_path, fi);
    printf("fclose(%s) input OK\n", input_path);

    FILE* f1 = open_input_stdio(input_path);
    printf("fopen(%s) input 1 OK\n", input_path);
    FILE* f2 = open_input_stdio(input_path);
    printf("fopen(%s) input 2 OK\n", input_path);
    close_stdio(input_path, f1);
    printf("fclose(%s) input 1 OK\n", input_path);
    close_stdio(input_path, f2);
    printf("fclose(%s) input 2 OK\n", input_path);
}

static void open_close_output_fd(const char* output_path) {
    int fo = open_output_fd(output_path, /*rdwr=*/false);
    printf("open(%s) output OK\n", output_path);
    close_fd(output_path, fo);
    printf("close(%s) output OK\n", output_path);

    int f1 = open_output_fd(output_path, /*rdwr=*/false);
    printf("open(%s) output 1 OK\n", output_path);
    int f2 = open_output_fd(output_path, /*rdwr=*/false);
    printf("open(%s) output 2 OK\n", output_path);
    close_fd(output_path, f1);
    printf("close(%s) output 1 OK\n", output_path);
    close_fd(output_path, f2);
    printf("close(%s) output 2 OK\n", output_path);
}

static void open_close_output_stdio(const char* output_path) {
    FILE* fo = open_output_stdio(output_path, /*rdwr=*/false);
    printf("fopen(%s) output OK\n", output_path);
    close_stdio(output_path, fo);
    printf("fclose(%s) output OK\n", output_path);

    FILE* f1 = open_output_stdio(output_path, /*rdwr=*/false);
    printf("fopen(%s) output 1 OK\n", output_path);
    FILE* f2 = open_output_stdio(output_path, /*rdwr=*/false);
    printf("fopen(%s) output 2 OK\n", output_path);
    close_stdio(output_path, f1);
    printf("fclose(%s) output 1 OK\n", output_path);
    close_stdio(output_path, f2);
    printf("fclose(%s) output 2 OK\n", output_path);
}

int main(int argc, char* argv[]) {
    if (argc < 3)
        fatal_error("Usage: %s {R|W} <path>\n", argv[0]);

    setup();

    if (toupper(argv[1][0]) == 'R') {
        open_close_input_fd(argv[2]);
        open_close_input_stdio(argv[2]);
    } else {
        open_close_output_fd(argv[2]);
        open_close_output_stdio(argv[2]);
    }

    return 0;
}
