uthash_h = custom_target('uthash.h',
    command: [
        patch_prog,
        '-p1',
        '--quiet',
        '--directory=@CURRENT_SOURCE_DIR@',
        '--input=uthash.h.patch',
        '--output=-',
    ],
    input: ['uthash.h.patch', 'uthash.h'],
    output: 'uthash.h',
    capture: true,
)
