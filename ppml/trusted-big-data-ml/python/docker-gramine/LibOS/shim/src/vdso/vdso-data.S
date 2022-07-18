    .section .rodata

    .global vdso_so
    .type vdso_so, @object
    .align 4
vdso_so:
    .incbin VDSO_SO_FULL_PATH

    .global vdso_so_end
    .type vdso_so_end, @object
    .align 4
vdso_so_end:

    .global vdso_so_size
    .type vdso_so_size, @object
    .size vdso_so_size, 8
    .align 8
vdso_so_size:
    .quad vdso_so_end - vdso_so
