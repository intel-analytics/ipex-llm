#include <err.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define STAT_FILE "/proc/stat"
#define BUFFSIZE  2048
#define KEYSIZE   32

/* see `man proc`, "/proc/stat" section */
struct procstat {
    uint64_t user;
    uint64_t nice;
    uint64_t system;
    uint64_t idle;
    uint64_t iowait;
    uint64_t irq;
    uint64_t softirq;
    uint64_t steal;
    uint64_t guest;
    uint64_t guest_nice;
};

static bool seen_ctxt          = false;
static bool seen_btime         = false;
static bool seen_processes     = false;
static bool seen_procs_running = false;
static bool seen_procs_blocked = false;

static void init_procstat(struct procstat* ps) {
    ps->user       = UINT64_MAX;
    ps->nice       = UINT64_MAX;
    ps->system     = UINT64_MAX;
    ps->idle       = UINT64_MAX;
    ps->iowait     = UINT64_MAX;
    ps->irq        = UINT64_MAX;
    ps->softirq    = UINT64_MAX;
    ps->steal      = UINT64_MAX;
    ps->guest      = UINT64_MAX;
    ps->guest_nice = UINT64_MAX;
}

static int check_procstat(struct procstat* ps) {
    if (ps->user == UINT64_MAX) {
        fprintf(stderr, "Could not get 'user' time\n");
        return -1;
    }
    if (ps->nice == UINT64_MAX) {
        fprintf(stderr, "Could not get 'nice' time\n");
        return -1;
    }
    if (ps->system == UINT64_MAX) {
        fprintf(stderr, "Could not get 'system' time\n");
        return -1;
    }
    if (ps->idle == UINT64_MAX) {
        fprintf(stderr, "Could not get 'idle' time\n");
        return -1;
    }
    if (ps->iowait == UINT64_MAX) {
        fprintf(stderr, "Could not get 'iowait' time\n");
        return -1;
    }
    if (ps->irq == UINT64_MAX) {
        fprintf(stderr, "Could not get 'irq' time\n");
        return -1;
    }
    if (ps->softirq == UINT64_MAX) {
        fprintf(stderr, "Could not get 'softirq' time\n");
        return -1;
    }
    if (ps->steal == UINT64_MAX) {
        fprintf(stderr, "Could not get 'steal' time\n");
        return -1;
    }
    if (ps->guest == UINT64_MAX) {
        fprintf(stderr, "Could not get 'guest' time\n");
        return -1;
    }
    if (ps->guest_nice == UINT64_MAX) {
        fprintf(stderr, "Could not get 'guest_nice' time\n");
        return -1;
    }
    return 0;
}

static int parse_and_check_noncpu_line(char* line) {
    int ret;

#define CHECK_LINE(seen_var, fmt)                                          \
    do {                                                                   \
        uint64_t val = UINT64_MAX;                                         \
        ret = sscanf(line, fmt, &val);                                     \
        if (ret < 0) {                                                     \
            fprintf(stderr, "cannot parse '%s'\n", line);                  \
            return -1;                                                     \
        }                                                                  \
        if (ret == 1) {                                                    \
            if (seen_var) {                                                \
                fprintf(stderr, "saw line '%s' more than once\n", line);   \
                return -1;                                                 \
            }                                                              \
            if (val == UINT64_MAX) {                                       \
                fprintf(stderr, "found wrong value in line '%s'\n", line); \
                return -1;                                                 \
            }                                                              \
            seen_var = true;                                               \
            return 0;                                                      \
        }                                                                  \
    } while (0)

    CHECK_LINE(seen_ctxt,          "ctxt %lu\n");
    CHECK_LINE(seen_btime,         "btime %lu\n");
    CHECK_LINE(seen_processes,     "processes %lu\n");
    CHECK_LINE(seen_procs_running, "procs_running %lu\n");
    CHECK_LINE(seen_procs_blocked, "procs_blocked %lu\n");
#undef CHECK_LINE

    fprintf(stderr, "found unrecognized line '%s'\n", line);
    return -1;
}

int main(int argc, char* argv[]) {
    FILE* fp = NULL;
    char line[BUFFSIZE];
    char cpu[KEYSIZE];
    struct procstat ps;
    int cpu_idx, cpu_cnt = 0, rv = 0;

    long actual_cpu_cnt = sysconf(_SC_NPROCESSORS_ONLN);
    if (actual_cpu_cnt < 0)
        errx(1, "cannot retrieve number of CPUs");

    if ((fp = fopen(STAT_FILE, "r")) == NULL)
        err(1, "fopen");

    /* first line is "cpu" (system-wide stats on times) */
    init_procstat(&ps);

    if (fgets(line, sizeof(line), fp) == NULL)
        errx(1, "cannot read 'cpu' line");

    rv = sscanf(line, "%s %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu\n", cpu, &ps.user, &ps.nice,
                &ps.system, &ps.idle, &ps.iowait, &ps.irq, &ps.softirq, &ps.steal, &ps.guest,
                &ps.guest_nice);
    if (rv != 11)
        errx(1, "cannot parse 'cpu' line");

    if (strcmp(cpu, "cpu"))
        errx(1, "did not find 'cpu' line");

    if ((rv = check_procstat(&ps)) != 0)
        errx(1, "unexpected values in 'cpu' line");

    /* next lines are "cpuX" (per-CPU stats on times) */
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (memcmp(line, "cpu", sizeof("cpu") - 1))
            break;

        init_procstat(&ps);
        rv = sscanf(line, "cpu%d %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu\n", &cpu_idx, &ps.user,
                    &ps.nice, &ps.system, &ps.idle, &ps.iowait, &ps.irq, &ps.softirq, &ps.steal,
                    &ps.guest, &ps.guest_nice);
        if (rv != 11)
            errx(1, "cannot parse 'cpu%d' line", cpu_cnt);

        if (cpu_idx != cpu_cnt)
            errx(1, "unexpected 'cpu%d' line (expected 'cpu%d')", cpu_idx, cpu_cnt);

        if ((rv = check_procstat(&ps)) != 0)
            errx(1, "unexpected values in 'cpu%d' line", cpu_cnt);
        cpu_cnt++;
    }

    if (cpu_cnt != actual_cpu_cnt)
        errx(1, "expected to find %ld CPUs but found only %d CPUs", actual_cpu_cnt, cpu_cnt);

    /* next lines are 'ctxt', 'btime', 'processes', 'procs_running', 'procs_blocked' */
    do {
        if (line[0] == '\n')
            break;
        if ((rv = parse_and_check_noncpu_line(line)) != 0)
            errx(1, "checking non-cpu line failed");
    } while (fgets(line, sizeof(line), fp) != NULL);

    if (!seen_ctxt || !seen_btime || !seen_processes || !seen_procs_running || !seen_procs_blocked)
        errx(1, "did not find ctxt/btime/processes/procs_running/procs_blocked line(s)");

    fclose(fp);

    printf("/proc/stat test passed\n");
    return 0;
}
