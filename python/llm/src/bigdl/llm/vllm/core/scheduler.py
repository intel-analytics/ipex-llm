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
# Some parts of this file is adapted from
# https://github.com/vllm-project/vllm/blob/v0.2.1.post1/vllm/core/scheduler.py
# which is licensed under Apache License 2.0
#
# Copyright 2023 The vLLM team. All rights reserved.
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
# bigdl-llm Intel specified code change
#

import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

from bigdl.llm.vllm.config import SchedulerConfig
from bigdl.llm.vllm.core.policy import PolicyFactory
from bigdl.llm.vllm.logger import init_logger
from bigdl.llm.vllm.sequence import SequenceData, SequenceStatus
from bigdl.llm.vllm.sequence import (Sequence, SequenceGroup,
                                     SequenceGroupMetadata)
from bigdl.llm.utils.common import invalidInputError

logger = init_logger(__name__)


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        ignored_seq_groups: List[SequenceGroup],
        finished_seqs: List[int],
    ) -> None:
        # bigdl-llm change start
        # Summary: we are removing block table related arguments
        # We also added finished_seqs so that workers know which sequences
        # can be safely deleted
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.ignored_seq_groups = ignored_seq_groups
        self.finished_seqs = finished_seqs
        # bigdl-llm change end

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.finished_seqs)


class FixedWindowScheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # bigdl-llm change start
        # summary: cache_config is removed as we are not implementing the pagetable structure
        # block manager is also deleted based on the same reasoning.
        # As we are not using the pagetable, the swapped area is also deleted because
        # we cannot decide if there is enough memory or not.

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")

        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        self.cleaned: List[int] = []
        # Co(gc): We no longer have the swapped space as we are not deciding which to swap
        # bigdl-llm change end

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running)

    def _schedule(self) -> SchedulerOutputs:

        # Fix the current time.
        now = time.monotonic()
        ignored_seq_groups: List[SequenceGroup] = []
        scheduled: List[SequenceGroup] = []
        finished_seqs: List[int] = self.cleaned.copy()
        self.cleaned = []
        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in self.running)
        num_batched_tokens = 0

        if self.waiting:
            # We restrict how many requests that can be run using these three arguments
            # Co(gc): If there are waiting requests, we will just try to add it into the
            # running state if not exceeds the stage
            # Co(gc): prefilled requests are prioritized over decoding stage requests
            while self.waiting:
                seq_group = self.waiting[0]

                invalidInputError(seq_group.num_seqs() == 1,
                                  "Waiting sequence group should have only one prompt "
                                  "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # bigdl-llm change start
                # summary: removing block_manager related logic.
                # TODO(gc): If you can manage to make block_manager work,
                #  then this will be fine.
                # If the sequence group cannot be allocated, stop.
                # if not self.block_manager.can_allocate(seq_group):
                #     break
                # bigdl-llm change end

                # If the number of batched tokens exceeds the limit, stop.
                if (num_batched_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.waiting.pop(0)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.RUNNING
                # bigdl-llm change start
                # summary: removing block_manager related logic.
                # self._allocate(seq_group)
                # bigdl-llm change end
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            scheduler_outputs = SchedulerOutputs(
                scheduled_seq_groups=scheduled,
                prompt_run=True,
                num_batched_tokens=num_batched_tokens,
                ignored_seq_groups=ignored_seq_groups,
                finished_seqs=finished_seqs,
            )
            return scheduler_outputs

        # Now consider all the requests in decoding stage
        self.running = self.policy.sort_by_priority(now, self.running)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            ignored_seq_groups=[],
            finished_seqs=finished_seqs,
        )
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, List[SequenceData]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def free_seq(self, seq: Sequence) -> None:
        # bigdl-llm specified change
        # bigdl-llm change start
        # summary: The original code free the block in block_manager.
        # now, we added it into a list to pass to worker in the next model_execute stage.
        self.cleaned.append(seq.seq_id)
        # bigdl-llm change end

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]
