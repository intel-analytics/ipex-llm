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

from ..nn.fl_client import FLClient


def init_fl_context(client_id: int, server_addr: str="localhost:8980") -> None:
    """Initialize FL Context. Need to be called before calling any FL Client algorithms.
    
    :param client_id: An integer, should be in range of [1, total_party_number].
    :param server_addr: FL Server address.
    """
    FLClient.load_config()
    FLClient.set_client_id(client_id)
    # target can be set in config file, and also could be overwritten here
    FLClient.set_target(server_addr)
    FLClient.ensure_initialized()
