#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import platform
from packaging import version
from importlib.metadata import distribution, PackageNotFoundError
from ipex_llm.utils.common import log4Error


class GlibcChecker:
    def __init__(self, min_glibc_version):
        self.min_glibc_version = min_glibc_version

    @staticmethod
    def is_linux():
        return platform.system() == "Linux"

    @staticmethod
    def is_bigdl_core_xe_installed():
        # Check if xpu version installed
        try:
            distribution('bigdl-core-xe')
            return True
        except PackageNotFoundError:
            return False

    @staticmethod
    def get_glibc_version():
        glibc_version_str = os.confstr('CS_GNU_LIBC_VERSION').split()[1]
        return version.parse(glibc_version_str)

    def check_requirements(self):
        if self.is_linux() and not self.is_bigdl_core_xe_installed():
            # Only report error for CPU
            glibc_version = self.get_glibc_version()
            if glibc_version < version.parse(self.min_glibc_version):
                log4Error.invalidInputError(
                    glibc_version >= version.parse(self.min_glibc_version),
                    f"Detected glibc version: {glibc_version}. "
                    f"(required: >= {self.min_glibc_version}) "
                    "Please upgrade your operating system with a newer version of glibc."
                )


glibc_checker = GlibcChecker("2.17")


def check_glibc_version():
    glibc_checker.check_requirements()
