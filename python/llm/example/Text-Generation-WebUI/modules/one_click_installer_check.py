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

# This file is adapted from
# https://github.com/oobabooga/text-generation-webui/blob/main/modules/one_click_installer_check.py


from pathlib import Path

from modules.logging_colors import logger

if Path('../webui.py').exists():
    logger.warning('\nIt looks like you are running an outdated version of '
                   'the one-click-installers.\n'
                   'Please migrate your installation following the instructions here:\n'
                   'https://github.com/oobabooga/text-generation-webui/wiki/Migrating-an-old-one%E2%80%90click-install')
