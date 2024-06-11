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

from importlib.metadata import distribution, PackageNotFoundError
import logging
import builtins
import sys
from ipex_llm.utils.common import log4Error
import inspect

# Save the original __import__ function
original_import = builtins.__import__
ipex_duplicate_import_error = "intel_extension_for_pytorch has already been automatically " + \
    "imported. Please avoid importing it again!"


def get_calling_package():
    """
    Return calling package name, e.g., ipex_llm.transformers
    """
    # Get the current stack frame
    frame = inspect.currentframe()
    # Get the caller's frame
    caller_frame = frame.f_back.f_back
    # Get the caller's module
    module = inspect.getmodule(caller_frame)
    if module:
        # Return the module's package name
        return module.__package__
    return None


def custom_ipex_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Custom import function to avoid importing ipex again
    """
    # check import calling pacage
    calling_package = get_calling_package()
    if calling_package is not None:
        return original_import(name, globals, locals, fromlist, level)
    # Only check ipex for main thread
    if name == "ipex" or name == "intel_extension_for_pytorch":
        log4Error.invalidInputError(False,
                                    ipex_duplicate_import_error)
    return original_import(name, globals, locals, fromlist, level)


class IPEXImporter:
    """
    Auto import Intel Extension for PyTorch as ipex,
    if bigdl-llm xpu version is installed.
    """
    def __init__(self):
        self.ipex_version = None

    @staticmethod
    def is_xpu_version_installed():
        """
        Check if bigdl-llm xpu version is install

        Returns ture if installed false if not
        """
        # Check if xpu version installed
        try:
            # Check if bigdl-core-xe is installed
            distribution('bigdl-core-xe')
            return True
        except PackageNotFoundError:
            # bigdl-core-xe not found
            # Check if bigdl-core-xe-21 is installed
            try:
                distribution('bigdl-core-xe-21')
                return True
            except PackageNotFoundError:
                # bigdl-core-xe not found
                return False
        return False

    def import_ipex(self):
        """
        Try to import Intel Extension for PyTorch as ipex for XPU

        Raises ImportError and invalidInputError if failed
        """
        if self.is_xpu_version_installed():
            # Check if user import ipex manually
            if 'ipex' in sys.modules or 'intel_extension_for_pytorch' in sys.modules:
                log4Error.invalidInputError(False,
                                            ipex_duplicate_import_error)
            self.directly_import_ipex()
            self.ipex_version = ipex.__version__
            # Replace default importer
            builtins.__import__ = custom_ipex_import
            logging.info("intel_extension_for_pytorch auto imported")

    def directly_import_ipex(self):
        """
        Try to import Intel Extension for PyTorch as ipex

        Raises ImportError and invalidInputError if failed
        """
        # import ipex
        import intel_extension_for_pytorch as ipex
        if ipex is not None:
            # Expose ipex to Python builtins
            builtins.ipex = ipex
        else:
            log4Error.invalidInputError(False,
                                        "Can not import intel_extension_for_pytorch.")

    def get_ipex_version(self):
        """
        Get ipex version

        Raises ImportError if cannot import Intel Extension for PyTorch
        """
        if self.ipex_version is not None:
            return self.ipex_version
        # try to import Intel Extension for PyTorch and get version
        self.directly_import_ipex()
        self.ipex_version = ipex.__version__
        # Replace default importer
        builtins.__import__ = custom_ipex_import
        return self.ipex_version


ipex_importer = IPEXImporter()
