/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2018 Intel Corporation
 *                    Isaku Yamahata <isaku.yamahata at gmail.com>
 *                                   <isaku.yamahata at intel.com>
 */

/*
 * Linker script for Gramine vDSO emulation
 */

PHDRS
{
        text            PT_LOAD         FLAGS(5) FILEHDR PHDRS;
        dynamic         PT_DYNAMIC      FLAGS(4);
        note            PT_NOTE         FLAGS(4);
        eh_frame_hdr    PT_GNU_EH_FRAME;
}

SECTIONS
{
        . = SIZEOF_HEADERS;
        .hash : { *(.hash) } :text
        .gnu.hash : { *(.gnu.hash) }
        .dynsym : { *(.dynsym) }
        .dynstr : { *(.dynstr) }
        .gnu.version : { *(.gnu.version) }
        .gnu.version_d : { *(.gnu.version_d) }
        .gnu.version_r : { *(.gnu.version_r) }
        .dynamic : { *(.dynamic) } :text :dynamic
        .rodata : {
                *(.rodata*)
                *(.data*)
                *(.sdata*)
                *(.got.plt) *(.got)
                *(.gnu.linkonce.d.*)
                *(.bss*)
                *(.dynbss*)
                *(.gnu.linkonce.b.*)
        } : text
        .note : { *(.note.*) } :text :note
        .eh_frame_hdr : { *(.eh_frame_hdr) } :text :eh_frame_hdr
        .eh_frame : { KEEP (*(.eh_frame)) } :text
        .text   : { *(.text*) } : text = 0x90909090,

        /DISCARD/ : {
                *(.discard)
                *(.discard.*)
        }
}

VERSION {
        LINUX_2.6 {
        global:
                clock_gettime;
                __vdso_clock_gettime;
                gettimeofday;
                __vdso_gettimeofday;
                getcpu;
                __vdso_getcpu;
                time;
                __vdso_time;

                __vdso_syscalldb;
        local: *;
        };
}
