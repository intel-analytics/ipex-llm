#include <dirent.h>
#include <stdio.h>

static int showdir(const char* path) {
    struct dirent* de;

    DIR* dir = opendir(path);
    if (!dir) {
        printf("Could not open directory `%s`\n", path);
        return 1;
    }

    printf("Contents of directory `%s`:\n", path);
    while ((de = readdir(dir)))
        printf("  %s\n", de->d_name);
    printf("\n");

    closedir(dir);
    return 0;
}

int main(int argc, char** argv) {
    if (showdir("/"))
        return 1;

    if (showdir("/var/"))
        return 1;

    puts("Test was successful");
    return 0;
}
