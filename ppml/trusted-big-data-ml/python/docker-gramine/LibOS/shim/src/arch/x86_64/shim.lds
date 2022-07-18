OUTPUT_FORMAT("elf64-x86-64", "elf64-x86-64", "elf64-x86-64")
OUTPUT_ARCH(i386:x86-64)
ENTRY(shim_start)

SECTIONS
{
  /* Read-only sections, merged into text segment; */
  __load_address = .;
  . = SEGMENT_START("text-segment", 0) + SIZEOF_HEADERS;
  .note.gnu.build-id : { *(.note.gnu.build-id) }
  .hash          : { *(.hash) }
  .gnu.hash      : { *(.gnu.hash) }
  .dynsym        : { *(.dynsym) }
  .dynstr        : { *(.dynstr) }
  .gnu.version   : { *(.gnu.version) }
  .gnu.version_d : { *(.gnu.version_d) }
  .gnu.version_r : { *(.gnu.version_r) }
  .rela.dyn       :
     {
      *(.rela.init)
      *(.rela.text .rela.text.* .rela.gnu.linkonce.t.*)
      *(.rela.fini)
      *(.rela.rodata .rela.rodata.* .rela.gnu.linkonce.r.*)
      *(.rela.data .rela.data.* .rela.gnu.linkonce.d.*)
      *(.rela.tdata .rela.tdata.* .rela.gnu.linkonce.td.*)
      *(.rela.tbss .rela.tbss.* .rela.gnu.linkonce.tb.*)
      *(.rela.ctors)
      *(.rela.dtors)
      *(.rela.got)
      *(.rela.bss .rela.bss.* .rela.gnu.linkonce.b.*)
      *(.rela.ldata .rela.ldata.* .rela.gnu.linkonce.l.*)
      *(.rela.lbss .rela.lbss.* .rela.gnu.linkonce.lb.*)
      *(.rela.lrodata .rela.lrodata.* .rela.gnu.linkonce.lr.*)
      *(.rela.ifunc)
    }
  .rela.plt      :
    {
      *(.rela.plt)
      *(.rela.iplt)
    }
  .init          : { *(.init) }
  .plt           : { *(.plt) *(.iplt) }
  .text :
  {
    /* the rest of text segments */
    *(.text .stub .text.*);
  }
  .fini          : { *(.fini) }
  .rodata :
  {
    /* the rest of rodata */
    *(.rodata .rodata.*)
    . = ALIGN(8);
    __cp_name = .;
    *(SORT(.cp_name.*));
    __cp_func = .;
    *(SORT(.cp_func.*));
    __rs_func = .;
    *(SORT(.rs_func.*));
  }
  .eh_frame_hdr  : { *(.eh_frame_hdr) }
  .eh_frame      : ONLY_IF_RO { *(.eh_frame) }

  /* now adjust the address for the data segment */
  . = DATA_SEGMENT_ALIGN (CONSTANT (MAXPAGESIZE), CONSTANT (COMMONPAGESIZE));
  .eh_frame      : ONLY_IF_RW { *(.eh_frame) }
  .ctors         : { *(.ctors) }
  .dtors         : { *(.dtors) }
  .jcr           : { *(.jcr) }
  .dynamic       : { *(.dynamic) }
  .got           : { *(.got) *(.igot) }
  .got.plt       : { *(.got.plt) *(.igot.plt) }
  .init_array    : {
    __init_array_start = .;
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) .init_array))
    __init_array_end = .;
  }
  . = DATA_SEGMENT_RELRO_END (0, .);
  .data :
  {
    /* the rest of data segment */
    *(.data .data.*);
    . = ALIGN(CONSTANT(COMMONPAGESIZE));
    __migratable = .;
    *(.migratable);
    __migratable_end = .;
    . = ALIGN(CONSTANT(COMMONPAGESIZE));
  }
  .bss            :
  {
    *(.bss .bss.*)
    *(COMMON)
  }
  __load_address_end = .;
  . = DATA_SEGMENT_END(.);
}

