    /* This .note section informs dynamic linker about vDSO */
    .section .note.Linux, "a", @note

    .balign 4
    .long   6       /* namesz */
    .long   4       /* descsz */
    .long   0       /* type */
    .string "Linux" /* name */
    .zero   2       /* padding for 4-byte alignment */
    .long   267008  /* LINUX_VERSION_CODE */

    .balign 4
    .long   6       /* namesz */
    .long   4       /* descsz */
    .long   0x100   /* type */
    .string "Linux" /* name */
    .zero   2       /* padding for 4-byte alignment */
    .long   0       /* CONFIG_BUILD_SALT. 0 for now.
                       TODO: make it compile-time configurable */
