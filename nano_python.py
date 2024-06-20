import os
import multiprocessing
import platform
import psutil
import re


#cores_per_socket

def del_env(env):
    #if env is not null, delete it
    #== unset in linux bash
    if os.getenv(env, 'null') != 'null':
        del os.environ[env]

def get_env(env):
    if os.environ.get(env):
        return os.environ[env]
    else:
        return ''

def get_whole_env(*argv):
    #args setting
    if 'disable-openmp' in argv:
        os.environ['DISABLE_OPENMP_VAR'] = "1"
    if 'enable-jemalloc' in argv:
        os.environ['ENABLE_JEMALLOC_VAR'] = "1"
    if 'disable-tcmalloc' in argv:
        os.environ['DISABLE_TCMALLOC_VAR'] = "1"
    if 'enable-tensorflow' in argv:
        os.environ['ENABLE_TF_OPTS'] = "1"

    # Init
    OPENMP = 0
    JEMALLOC = 0
    TCMALLOC = 0

    # Find conda dir
    BASE_DIR = os.getcwd()+'/..'
    LIB_DIR = BASE_DIR +'/lib'
    PYTHON_VERSION = platform.python_version()

    NANO_DIR = LIB_DIR + '/python' + PYTHON_VERSION + '/site-packages/bigdl/nano/'

    if os.path.isfile(LIB_DIR + '/libiomp5.so'):

        print("OpenMP library found...")
        OPENMP = 1

        # detect number of physical cores
        cpu_physical = psutil.cpu_count(logical=False)
        # detect number of total threads
        cpu_logical = psutil.cpu_count()
        cores_per_socket = multiprocessing.cpu_count()
        threads_per_core = cpu_logical/cpu_physical
        # how to get sockets_ in windows still unknown

        # set env variables
        print("Setting OMP_NUM_THREADS...")
        if os.getenv('ENABLE_TF_OPTS', 'null') == 'null':
            print("Setting OMP_NUM_THREADS specified for pytorch...")
            #                                   cores_per_socket*sockets_
            os.environ['OMP_NUM_THREADS'] = str(cpu_physical)
        else:
            #                                   cores_per_socket
            os.environ['OMP_NUM_THREADS'] = str(cpu_physical)

        print("Setting KMP_AFFINITY...")

        if (threads_per_core > 1):
            os.environ['KMP_AFFINITY'] = "granularity=fine,compact,1,0"
        else:
            os.environ['KMP_AFFINITY'] = "granularity=fine, compact"
        print("Setting KMP_BLOCKTIME...")
        os.environ['KMP_BLOCKTIME'] = str(1)
    else:
        print("No openMP library found in ", LIB_DIR, ".")

    # Detect jemalloc library
    JEMALLOC = 1

    # Detect tcmalloc library
    TCMALLOC = 1

    # set env variables
    print("Setting MALLOC_CONF...")
    os.environ['MALLOC_CONF'] = "oversize_threshold:1,background_thread:true,metadata_" \
                                "thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

    # Set LD_PRELOAD
    if os.getenv('LD_PRELOAD', 'null') == 'null':
        print("Setting LD_PRELOAD...")
        if OPENMP == 1 and os.getenv('DISABLE_OPENMP_VAR', 'null') == 'null':
            os.environ['LD_PRELOAD'] = LIB_DIR + '/libiomp5.so'

        if JEMALLOC == 1 and os.getenv('ENABLE_JEMALLOC_VAR', 'null') != 'null':
            os.environ['DISABLE_TCMALLOC_VAR'] = str(1)
            if os.getenv('LD_PRELOAD', 'null') == 'null':
                os.environ['LD_PRELOAD'] = LIB_DIR + '/libjemalloc.so'
            else:
                os.environ['LD_PRELOAD'] += " "+ NANO_DIR + \
                                           "/libs/libjemalloc.so"
        # Set TCMALLOC lib path
        if TCMALLOC == 1 and os.getenv('DISABLE_TCMALLOC_VAR', 'null') == 'null':
            if os.getenv('LD_PRELOAD', 'null') == 'null':
                os.environ['LD_PRELOAD'] = LIB_DIR + '/libtcmalloc.so'
            else:
                os.environ['LD_PRELOAD'] += " "+ NANO_DIR + \
                                           "/libs/libtcmalloc.so"
        # Set TF_ENABLE_ONEDNN_OPTS
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = str(1)

        # Disable openmp or jemalloc according to options
        if os.getenv('DISABLE_OPENMP_VAR', 'null') != 'null':
            del_env('OMP_NUM_THREADS')
            del_env('KMP_AFFINITY')
            del_env('KMP_BLOCKTIME')
            del_env('DISABLE_OPENMP_VAR')
            temp = os.environ['LD_PRELOAD']
            #delete content containing .libiomp5.so
            temp = re.sub(r'\s.*libiomp5\.so','',temp)
            os.environ['LD_PRELOAD'] = re.sub(r'.*libiomp5\.so\s*','',temp)

        if os.getenv('ENABLE_JEMALLOC_VAR', 'null') == 'null':
            #del_env('ENABLE_JEMALLOC_VAR')
            del_env('MALLOC_CONF')
            temp = os.environ['LD_PRELOAD']
            # delete content containing .libjemalloc.so
            temp = re.sub(r'\s.*libjemalloc\.so','',temp)
            os.environ['LD_PRELOAD'] = re.sub(r'.*libjemalloc\.so\s*','',temp)

        if os.getenv('DISABLE_TCMALLOC_VAR', 'null') != 'null':
            del_env('DISABLE_TCMALLOC_VAR')
            temp = os.environ['LD_PRELOAD']
            # delete content containing .libtcmalloc.so
            temp = re.sub(r'\s.*libtcmalloc\.so','',temp)
            os.environ['LD_PRELOAD'] = re.sub(r'.*libtcmalloc\.so\s*','',temp)

        # if os.getenv('LD_PRELOAD', 'null') == 'null':
        #     del_env('LD_PRELOAD')

        if os.getenv('ENABLE_TF_OPTS', 'null') == 'null':
            #del_env('ENABLE_TF_OPTS')
            del_env('TF_ENABLE_ONEDNN_OPTS')

        #Return dict of env_variables
        return {'LD_PRELOAD':get_env('LD_PRELOAD'),
                'MALLOC_CONF':get_env('MALLOC_CONF'),
                'OMP_NUM_THREADS':get_env('OMP_NUM_THREADS'),
                'KMP_AFFINITY':get_env('KMP_AFFINITY'),'KMP_BLOCKTIME':get_env('KMP_BLOCKTIME'),
                'TF_ENABLE_ONEDNN_OPTS':get_env('TF_ENABLE_ONEDNN_OPTS')}



#test
# dic_ =get_whole_env('enable-jemalloc')
# print(dic_)
