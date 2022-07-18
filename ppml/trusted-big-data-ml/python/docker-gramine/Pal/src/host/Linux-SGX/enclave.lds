OUTPUT_FORMAT("elf64-x86-64", "elf64-x86-64", "elf64-x86-64")
OUTPUT_ARCH(i386:x86-64)
ENTRY(enclave_entry)

SECTIONS
{
  /* Read-only sections, merged into text segment; */
  pal_start_addr = .;
  __text_start = .;
  . = SEGMENT_START("text-segment", 0) + SIZEOF_HEADERS;
  text_start = .;
  .note.gnu.build-id : { *(.note.gnu.build-id) }
  .gnu.hash      : { *(.gnu.hash) }
  .dynsym        : { *(.dynsym) }
  .dynstr        : { *(.dynstr) }
  .gnu.version   : { *(.gnu.version) }
  .gnu.version_d : { *(.gnu.version_d) }
  .rela.dyn      : { *(.rela.dyn) }
  .rela.plt      : { *(.rela.plt) }
  .init          : { *(.init) }
  .plt           : { *(.plt) *(.iplt) }
  .text :
  {
    g_section_text = .;
    /* the rest of text segments */
    *(.text .stub .text.*);
  }
  .fini          : { *(.fini) }
  .eh_frame_hdr  : { *(.eh_frame_hdr) }
  .eh_frame      : ONLY_IF_RO { KEEP(*(.eh_frame)) }
  .rodata :
  {
    g_section_rodata = .;
    /* the rest of rodata */
    *(.rodata .rodata.*)
  }
  __text_end = .;

  /* now adjust the address for the data segment */
  . = DATA_SEGMENT_ALIGN (CONSTANT (MAXPAGESIZE), CONSTANT (COMMONPAGESIZE));
 __data_start = .;
  .tdata         :
  {
    *(.tdata)
  } :tls

  .dynamic       :
  {
    g_section_dynamic = .;
    *(.dynamic)
  }

  data_start = .;
  .data.rel.ro   : { KEEP(*(.data.rel.ro)) }
  .ctors         : { KEEP(*(.ctors)) }
  .dtors         : { KEEP(*(.dtors)) }
  .jcr           : { KEEP(*(.jcr)) }
  .got           : { *(.got) *(.igot) }
  .got.plt       : { *(.got.plt) *(.igot.plt) }
  .init_array    : {
    __init_array_start = .;
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) .init_array))
    __init_array_end = .;
  }
  . = DATA_SEGMENT_RELRO_END (0, .);
  .data          :
  {
    g_section_data = .;
    /* the rest of data segment */
    *(.data .data.*);
  }
  .bss           :
  {
    g_section_bss = .;
    *(.bss)
    *(COMMON)
  }
  __data_end = .;

  . = DATA_SEGMENT_END(.);
}

