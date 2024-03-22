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
        Try to import Intel Extension for PyTorch as ipex

        Raises ImportError if failed
        """
        if self.is_xpu_version_installed():
            import intel_extension_for_pytorch as ipex
            self.ipex_version = ipex.__version__
            logging.info("intel_extension_for_pytorch auto imported")

    def get_ipex_version(self):
        """
        Get ipex version

        Raises ImportError if cannot import Intel Extension for PyTorch
        """
        if self.ipex_version is not None:
            return self.ipex_version
        # try to import Intel Extension for PyTorch and get version
        try:
            import intel_extension_for_pytorch as ipex
            self.ipex_version = ipex.__version__
        except ImportError:
            self.ipex_version = None
        return self.ipex_version


ipex_importer = IPEXImporter()
