#include "common.h"

#define EXTEND_SIZE 4097

static void seek_input_fd(const char* path) {
    int f = open_input_fd(path);
    printf("open(%s) input OK\n", path);
    seek_fd(path, f, 0, SEEK_SET);
    printf("seek(%s) input start OK\n", path);
    seek_fd(path, f, 0, SEEK_END);
    printf("seek(%s) input end OK\n", path);
    off_t pos = tell_fd(path, f);
    printf("tell(%s) input end OK: %zd\n", path, pos);
    seek_fd(path, f, -pos, SEEK_END); // rewind
    printf("seek(%s) input rewind OK\n", path);
    pos = tell_fd(path, f);
    printf("tell(%s) input start OK: %zd\n", path, pos);
    close_fd(path, f);
    printf("close(%s) input OK\n", path);
}

static void seek_input_stdio(const char* path) {
    FILE* f = open_input_stdio(path);
    printf("fopen(%s) input OK\n", path);
    seek_stdio(path, f, 0, SEEK_SET);
    printf("fseek(%s) input start OK\n", path);
    seek_stdio(path, f, 0, SEEK_END);
    printf("fseek(%s) input end OK\n", path);
    off_t pos = tell_stdio(path, f);
    printf("ftell(%s) input end OK: %zd\n", path, pos);
    seek_stdio(path, f, -pos, SEEK_END); // rewind
    printf("fseek(%s) input rewind OK\n", path);
    pos = tell_stdio(path, f);
    printf("ftell(%s) input start OK: %zd\n", path, pos);
    close_stdio(path, f);
    printf("fclose(%s) input OK\n", path);
}

static void seek_output_fd(const char* path) {
    uint8_t buf[EXTEND_SIZE + 1] = {1};
    int f = open_output_fd(path, /*rdwr=*/true);
    printf("open(%s) output OK\n", path);
    seek_fd(path, f, 0, SEEK_SET);
    printf("seek(%s) output start OK\n", path);
    seek_fd(path, f, 0, SEEK_END);
    printf("seek(%s) output end OK\n", path);
    off_t pos = tell_fd(path, f);
    printf("tell(%s) output end OK: %zd\n", path, pos);
    seek_fd(path, f, EXTEND_SIZE, SEEK_CUR); // extend
    printf("seek(%s) output end 2 OK\n", path);
    write_fd(path, f, buf, 1);
    seek_fd(path, f, -EXTEND_SIZE - 1, SEEK_CUR); // rewind to former end
    printf("seek(%s) output end 3 OK\n", path);
    read_fd(path, f, buf, EXTEND_SIZE + 1);
    for (size_t i = 0; i < EXTEND_SIZE + 1; i++) {
        if (i == EXTEND_SIZE) {
            if (buf[i] != 1)
                fatal_error("invalid last byte\n");
        } else {
            if (buf[i] != 0)
                fatal_error("extended buffer not zeroed\n");
        }
    }
    pos = tell_fd(path, f);
    printf("tell(%s) output end 2 OK: %zd\n", path, pos);
    close_fd(path, f);
    printf("close(%s) output OK\n", path);
}

static void seek_output_stdio(const char* path) {
    uint8_t buf[EXTEND_SIZE + 1] = {1};
    FILE* f = open_output_stdio(path, /*rdwr=*/true);
    printf("fopen(%s) output OK\n", path);
    seek_stdio(path, f, 0, SEEK_SET);
    printf("fseek(%s) output start OK\n", path);
    seek_stdio(path, f, 0, SEEK_END);
    printf("fseek(%s) output end OK\n", path);
    off_t pos = tell_stdio(path, f);
    printf("ftell(%s) output end OK: %zd\n", path, pos);
    seek_stdio(path, f, EXTEND_SIZE, SEEK_CUR); // extend
    printf("fseek(%s) output end 2 OK\n", path);
    write_stdio(path, f, buf, 1);
    seek_stdio(path, f, -EXTEND_SIZE - 1, SEEK_CUR); // rewind to former end
    printf("fseek(%s) output end 3 OK\n", path);
    read_stdio(path, f, buf, EXTEND_SIZE + 1);
    for (size_t i = 0; i < EXTEND_SIZE + 1; i++) {
        if (i == EXTEND_SIZE) {
            if (buf[i] != 1)
                fatal_error("invalid last byte\n");
        } else {
            if (buf[i] != 0)
                fatal_error("extended buffer not zeroed\n");
        }
    }
    pos = tell_stdio(path, f);
    printf("ftell(%s) output end 2 OK: %zd\n", path, pos);
    close_stdio(path, f);
    printf("fclose(%s) output OK\n", path);
}

int main(int argc, char* argv[]) {
    if (argc < 4)
        fatal_error("Usage: %s <input_path> <output_path_1> <output_path_2>\n", argv[0]);

    setup();
    seek_input_fd(argv[1]);
    seek_input_stdio(argv[1]);
    seek_output_fd(argv[2]);
    seek_output_stdio(argv[3]);

    return 0;
}
