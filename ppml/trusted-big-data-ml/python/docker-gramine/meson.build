#
# Project configuration, options, modules, scripts
#

project(
    'gramine',
    'c', 'cpp',
    version: '1.2post-UNRELEASED',
    license: 'LGPLv3+',

    meson_version: '>=0.55',

    default_options: [
        'c_std=c11',
        'cpp_std=c++14',
    ],
)

# we need this subdir() early, because we need scripts defined there for setting up global vars
subdir('Scripts')

prefix = get_option('prefix')
pkglibdir = join_paths(get_option('libdir'), meson.project_name())
pkgdatadir = join_paths(get_option('datadir'), meson.project_name())

syslibdir = get_option('syslibdir')
if syslibdir == ''
    syslibdir = join_paths('/', get_option('libdir'))
endif

direct = get_option('direct') == 'enabled'
sgx = get_option('sgx') == 'enabled'
vtune = get_option('vtune') == 'enabled'
skeleton = get_option('skeleton') == 'enabled'

dcap = get_option('dcap') == 'enabled'
ubsan = get_option('ubsan') == 'enabled'
asan = get_option('asan') == 'enabled'

enable_libgomp = get_option('libgomp') == 'enabled'
enable_tests = get_option('tests') == 'enabled'

enable_glibc = get_option('glibc') == 'enabled'
enable_musl = get_option('musl') == 'enabled'

cc = meson.get_compiler('c')
objcopy = find_program('objcopy')

# TODO: after deprecating 18.04/bionic, update this to import('python')
python3mod = import('python3')
python3 = python3mod.find_python()
python3_platlib = run_command(
    python3, get_python_platlib_prog, get_option('prefix')).stdout()
python3_pkgdir = join_paths(python3_platlib, 'graminelibos')

pkgconfig = import('pkgconfig')

if host_machine.cpu_family() == 'x86_64'
    nasm_gen = generator(
        find_program('nasm'),
        output: '@BASENAME@.o',
        depfile: '@BASENAME@.dep',
        arguments: [
            '-f', 'elf64',
            '-I', '@0@/'.format(meson.current_build_dir()),
            '-MQ', '@OUTPUT@', '-MF', '@DEPFILE@',
            '@EXTRA_ARGS@',
            '@INPUT@',
            '-o', '@OUTPUT@'
        ]
    )
endif

add_project_arguments(
    '-Wa,--noexecstack',

    '-Wall',
    '-Wextra',

    '-Wmissing-prototypes',
    '-Wstrict-prototypes',
    '-Wwrite-strings',

    cc.get_supported_arguments(
        '-Wtrampolines',
        '-Wnull-dereference',
    ),

    language: 'c')

debug = get_option('buildtype') == 'debug' or get_option('buildtype') == 'debugoptimized'
if debug
    add_project_arguments('-DDEBUG', language: 'c')
endif

if sgx
    conf_sgx = configuration_data()

    # SGX driver options (in Make, this was ISGX_DRIVER_PATH and link-intel-driver.py)
    sgx_driver = get_option('sgx_driver')
    sgx_driver_include_path = get_option('sgx_driver_include_path')
    sgx_driver_device = get_option('sgx_driver_device')

    if sgx_driver == 'upstream'
        # upstream in-kernel driver (Linux 5.11+)
        conf_sgx.set('CONFIG_SGX_DRIVER_UPSTREAM', true)
        sgx_driver_header = 'asm/sgx.h'
        sgx_driver_include_path_defaults = [
            '/usr/include/',
            # Ubuntu `linux-headers` package
            '/usr/src/linux-headers-@0@/arch/x86/include/uapi'.format(
                run_command('uname', '-r').stdout().strip()),
        ]
        sgx_driver_device_default = '/dev/sgx_enclave'
    elif sgx_driver == 'oot'
        # old non-DCAP driver (https://github.com/intel/linux-sgx-driver)
        conf_sgx.set('CONFIG_SGX_DRIVER_OOT', true)
        sgx_driver_header = 'sgx_user.h'
        sgx_driver_device_default = '/dev/isgx'
        sgx_driver_include_path_defaults = ['/opt/intel/linux-sgx-driver']
    elif sgx_driver == 'dcap1.6'
        # DCAP 1.6+ but below 1.10 (https://github.com/intel/SGXDataCenterAttestationPrimitives)
        conf_sgx.set('CONFIG_SGX_DRIVER_DCAP_1_6', true)
        sgx_driver_header = 'uapi/asm/sgx_oot.h'
        sgx_driver_device_default = '/dev/sgx/enclave'
        sgx_driver_include_path_defaults = [
            '/opt/intel/SGXDataCenterAttestationPrimitives/driver/linux/include',
        ]
    elif sgx_driver == 'dcap1.10'
        # DCAP 1.10+ (https://github.com/intel/SGXDataCenterAttestationPrimitives)
        conf_sgx.set('CONFIG_SGX_DRIVER_DCAP_1_10', true)
        sgx_driver_header = 'sgx_user.h'
        sgx_driver_device_default = '/dev/sgx/enclave'
        sgx_driver_include_path_defaults = [
            '/opt/intel/SGXDataCenterAttestationPrimitives/driver/linux/include',
        ]
    else
        error('Unknown sgx_driver value')
    endif

    if sgx_driver_device == ''
        sgx_driver_device = sgx_driver_device_default
    endif

    if sgx_driver_include_path == ''
        foreach path : sgx_driver_include_path_defaults
            sgx_driver_header_abspath = ''
            if cc.has_header(join_paths(path, sgx_driver_header))
                sgx_driver_include_path = path
                break
            endif
        endforeach

        if sgx_driver_include_path == ''
            error('Invalid SGX driver configuration (-Dsgx_driver and/or ' +
                  '-Dsgx_driver_include_path); searched for "@0@" under directories: @1@'.format(
                      sgx_driver_header, sgx_driver_include_path_defaults))
        endif
    endif


    sgx_driver_header_abspath = join_paths(sgx_driver_include_path, sgx_driver_header)

    if not cc.has_header(sgx_driver_header_abspath)
        error('Invalid SGX driver configuration (-Dsgx_driver and/or -Dsgx_driver_include_path); ' +
              'expected "@0@" to exist under "@1@"'.format(
                  sgx_driver_header, sgx_driver_include_path))
    endif

  if vtune
    add_project_arguments('-DSGX_VTUNE_PROFILE', language: 'c')
    vtune_sdk_path = get_option('vtune_sdk_path')
  endif

    conf_sgx.set('CONFIG_SGX_DRIVER_HEADER_ABSPATH', sgx_driver_header_abspath)
    conf_sgx.set_quoted('CONFIG_SGX_DRIVER_DEVICE', sgx_driver_device)
endif

gen_symbol_map_cmd = [
    find_program('nm'),
    '--numeric-sort', '--defined-only', '--print-size', '--line-numbers',
    '@INPUT@',
]

#
# Common checks and flags
#

# Not all compilers support mstack-protector-guard, so use stack protector only if supported.
# Gramine-custom stack protector uses the canary stored in the TCB (same for both in LibOS and PAL)
# at offset 0x8.
if host_machine.cpu_family() == 'x86_64'
    cflags_custom_stack_protector = [
        '-fstack-protector-strong',
        '-mstack-protector-guard=tls',
        '-mstack-protector-guard-reg=%gs',
        '-mstack-protector-guard-offset=8',
    ]
else
    cflags_custom_stack_protector = [
        '-fstack-protector-strong',
    ]
endif

if not cc.has_multi_arguments(cflags_custom_stack_protector)
    cflags_custom_stack_protector = '-fno-stack-protector'
endif

# Don't support b_sanitize: integration with sanitizers in Gramine is tricky and we want more
# control over the exact flags.
if get_option('b_sanitize') != 'none'
    error('Please don\'t use the b_sanitize option; use -Dubsan=enabled instead.')
endif

cflags_sanitizers = []

if ubsan
    cflags_sanitizers += [
        '-fsanitize=undefined',
        '-fno-sanitize-recover=undefined',
        '-DUBSAN',
    ]
endif

if asan
    if meson.get_compiler('c').get_id() != 'clang' or meson.get_compiler('cpp').get_id() != 'clang'
        error('ASan is currently supported only for Clang (CC=clang CXX=clang++)')
    endif

    asan_shadow_start = '0x18000000000'
    cflags_sanitizers += [
        '-fsanitize=address',
        '-fno-sanitize-link-runtime',
        '-mllvm', '-asan-mapping-offset=' + asan_shadow_start,
        '-DASAN',
    ]

    if meson.get_compiler('c').version().version_compare('>=13')
        cflags_sanitizers += ['-mllvm', '-asan-use-after-return=never']
    else
        cflags_sanitizers += ['-mllvm', '-asan-use-after-return=0']
    endif
endif

#
# Dependencies
#

tomlc99_proj = subproject('tomlc99-208203af46bdbdb29ba199660ed78d09c220b6c5')
tomlc99_dep = tomlc99_proj.get_variable('tomlc99_dep')
tomlc99_src = tomlc99_proj.get_variable('tomlc99_src')

uthash_dep = subproject('uthash-2.1.0').get_variable('uthash_dep')

mbedtls_proj = subproject('mbedtls-mbedtls-2.26.0-2')
mbedtls_static_dep = mbedtls_proj.get_variable('mbedtls_static_dep')
mbedtls_pal_dep = mbedtls_proj.get_variable('mbedtls_pal_dep')

if sgx
    protoc_c_prog = find_program('protoc-c')

    threads_dep = dependency('threads')

    libcurl_dep = dependency('libcurl', version: '>=7.58.0')

    cjson_dep = dependency('cjson', required: false)
    if not cjson_dep.found()
        # on Debian < 11 .pc file is not installed and dependency() fails, so we fallback to
        # cc.find_library()
        cjson_dep = cc.find_library('cjson', required: false)
    endif
    if not cjson_dep.found()
        # Ubuntu 18.04 does not have cjson at all
        cjson_dep = subproject('cJSON-1.7.12').get_variable('cjson_dep')
    endif

    protobuf_dep = dependency('libprotobuf-c')

    if dcap
        sgx_dcap_quoteverify_dep = cc.find_library('sgx_dcap_quoteverify')
    endif

  vtune_dep = []
  dl_dep = []

  if vtune
      dl_dep = cc.find_library('dl', required: true)
      vtune_dep = declare_dependency(
          dependencies: [
              cc.find_library('ittnotify', dirs: join_paths(vtune_sdk_path, 'lib64'), required: true),
              dl_dep,
          ],
          include_directories: include_directories(join_paths(vtune_sdk_path, 'include')),
      )
  endif

endif

#
# The compilation
#

subdir('common')
subdir('Pal')
subdir('LibOS')
subdir('python')
subdir('tools')

if enable_glibc
    subproject('glibc-2.34-1')
endif
if enable_musl
    subproject('musl-1.2.2')
endif

if enable_libgomp
    subproject('gcc-10.2.0')
endif

run_target('clang-format', command: [meson_clang_format_prog])
