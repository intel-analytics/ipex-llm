import sys
from bigdl.llm.utils.common import invalidInputError, invalidOperationError

def get_avx_flags():
    avx = ""
    if sys.platform != "win32":
        import subprocess
        msg = subprocess.check_output(["lscpu"]).decode("utf-8")
        if "avx512_vnni" in msg:
            avx = "_avx512"
        elif "avx2" in msg:
            avx = "_avx2"
        else:
            invalidOperationError(False, "Unsupported CPUFLAGS.")
    return avx
