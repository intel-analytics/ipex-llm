#include <stdio.h>
#include <string.h>

#define CPUINFO_FILE "/proc/cpuinfo"
#define BUFFSIZE     2048

/* vendor_id, model_name size reference Linux kernel struct cpuinfo_x86
 * (see Linux's arch/x86/include/asm/processor.h) */
struct cpuinfo {
    int processor;
#if defined(__i386__) || defined(__x86_64__)
    char vendor_id[16];
    int cpu_family;
    int model;
    char model_name[64];
    int stepping;
    int core_id;
    int cpu_cores;
#endif
};

static void init_cpuinfo(struct cpuinfo* ci) {
    ci->processor = -1;
#if defined(__i386__) || defined(__x86_64__)
    memset(&ci->vendor_id, 0, sizeof(ci->vendor_id));
    ci->cpu_family = -1;
    ci->model      = -1;
    memset(&ci->model_name, 0, sizeof(ci->model_name));
    ci->stepping  = -1;
    ci->core_id   = -1;
    ci->cpu_cores = -1;
#endif
}

static int parse_line(char* line, struct cpuinfo* ci) {
    char *k, *v, *p;

    if ((p = strchr(line, ':')) == NULL)
        goto fmt_err;

    /* if the line does not have value string, p[1] should be '\n', otherwise
     * p[1] should be ' ' */
    if (p[1] == '\n' && !p[2])
        return 0; /* No value string */

    /* ':' should always be followed by a space */
    if (p[1] != ' ')
        goto fmt_err;

    /* skip ": " to get value string */
    v = p + 2;

    /* get key string */
    *p = '\0';
    if ((p = strchr(line, '\t')) != NULL)
        *p = '\0';
    k = line;

    if (!strcmp(k, "processor")) {
        sscanf(v, "%d\n", &ci->processor);
#if defined(__i386__) || defined(__x86_64__)
    } else if (!strcmp(k, "cpu family")) {
        sscanf(v, "%d\n", &ci->cpu_family);
    } else if (!strcmp(k, "model")) {
        sscanf(v, "%d\n", &ci->model);
    } else if (!strcmp(k, "stepping")) {
        sscanf(v, "%d\n", &ci->stepping);
    } else if (!strcmp(k, "core id")) {
        sscanf(v, "%d\n", &ci->core_id);
    } else if (!strcmp(k, "cpu cores")) {
        sscanf(v, "%d\n", &ci->cpu_cores);
    } else if (!strcmp(k, "vendor_id")) {
        snprintf(ci->vendor_id, sizeof(ci->vendor_id), "%s", v);
    } else if (!strcmp(k, "model name")) {
        snprintf(ci->model_name, sizeof(ci->model_name), "%s", v);
#endif
    }
    return 0;

fmt_err:
    fprintf(stderr, "format error in line: %s\n", line);
    return -1;
};

static int check_cpuinfo(struct cpuinfo* ci) {
    if (ci->processor == -1) {
        fprintf(stderr, "Could not get cpu index\n");
        return -1;
    }
#if defined(__i386__) || defined(__x86_64__)
    if (ci->core_id == -1) {
        fprintf(stderr, "Could not get core id\n");
        return -1;
    }
    if (ci->cpu_cores == -1) {
        fprintf(stderr, "Could not get cpu cores\n");
        return -1;
    }
#endif

    return 0;
}

int main(int argc, char* argv[]) {
    FILE* fp = NULL;
    char line[BUFFSIZE];
    struct cpuinfo ci;
    int cpu_cnt = 0, rv = 0;

    init_cpuinfo(&ci);

    if ((fp = fopen(CPUINFO_FILE, "r")) == NULL) {
        perror("fopen");
        return 1;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (line[0] == '\n') {
            if ((rv = check_cpuinfo(&ci)) != 0)
                break;
            cpu_cnt++;
            init_cpuinfo(&ci);
            continue;
        }
        if ((rv = parse_line(line, &ci)) != 0)
            break;
    }

    fclose(fp);

    if (rv != 0)
        return 1;

    if (cpu_cnt == 0) {
        fprintf(stderr, "Could not get online cpu info.\n");
        return 1;
    }

    printf("cpuinfo test passed\n");
    return 0;
}
