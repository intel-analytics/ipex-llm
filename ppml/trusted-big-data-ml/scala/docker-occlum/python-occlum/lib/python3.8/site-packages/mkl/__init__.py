# Copyright (c) 2018-2019, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys

class RTLD_for_MKL():
    def __init__(self):
        self.saved_rtld = None

    def __enter__(self):
        import ctypes
        try:
            self.saved_rtld = sys.getdlopenflags()
            # python loads libraries with RTLD_LOCAL, but MKL requires RTLD_GLOBAL
            # pre-load MKL with RTLD_GLOBAL before loading the native extension
            sys.setdlopenflags(self.saved_rtld | ctypes.RTLD_GLOBAL)
        except AttributeError:
            pass
        del ctypes

    def __exit__(self, *args):
        if self.saved_rtld:
            sys.setdlopenflags(self.saved_rtld)
            self.saved_rtld = None

with RTLD_for_MKL():
    from . import _mklinit

del RTLD_for_MKL
del sys

from ._py_mkl_service import *


__version__ = '2.4.0'
