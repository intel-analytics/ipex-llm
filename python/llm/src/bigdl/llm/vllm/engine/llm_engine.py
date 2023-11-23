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
# https://github.com/vllm-project/vllm/blob/v0.2.1.post1/vllm/engine/llm_engine.py
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
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

from bigdl.llm.vllm.config import ModelConfig, SchedulerConfig
from bigdl.llm.vllm.core.scheduler import SchedulerOutputs, FixedWindowScheduler
from bigdl.llm.vllm.engine.arg_utils import EngineArgs
from bigdl.llm.vllm.logger import init_logger
from bigdl.llm.vllm.outputs import RequestOutput
from bigdl.llm.vllm.sampling_params import SamplingParams
from bigdl.llm.vllm.sequence import (
    SamplerOutput,
    Sequence,
    SequenceGroup,
    SequenceGroupMetadata,
    SequenceStatus,
    SequenceOutputs,
)
from bigdl.llm.vllm.transformers_utils.tokenizer import get_tokenizer, detokenize_incrementally
from bigdl.llm.vllm.utils import (
    Counter,
)
from bigdl.llm.utils.common import invalidInputError

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        # parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        # distributed_init_method: str,
        # placement_group,
        log_stats: bool,
    ) -> None:
        # bigdl-llm change start
        # summary: removing parallel_config and related checks.
        # distributed_init_method/placement_group is related to these configs
        # so they are removed too.
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            # f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"seed={model_config.seed})"
        )
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        # self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        # self._verify_args()

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            tokenizer_revision=model_config.tokenizer_revision,
            revision=model_config.revision,
        )
        self.seq_counter = Counter()

        # Create the parallel GPU workers.
        self._init_workers()

        # Co(gc): we create a fixed scheduler
        self.scheduler = FixedWindowScheduler(scheduler_config)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []
        # bigdl-llm change end

    def _init_workers(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from bigdl.llm.vllm.worker.worker import (
            Worker,
        )  # pylint: disable=import-outside-toplevel

        # invalidInputError(
        #     self.parallel_config.world_size == 1,
        #     "Ray is required if parallel_config.world_size > 1.",
        # )

        self.workers: List[Worker] = []
        worker = Worker(
            self.model_config,
            self.scheduler_config,
            0,
            # distributed_init_method,
        )
        self.workers.append(worker)
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        # Co(gc): this simply checks if the swap is too large or not
        # self.cache_config.verify_with_parallel_config(self.parallel_config)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # bigdl-llm change start
        # summary: remove parallel_config and related settings.
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        # parallel_config = engine_configs[2]
        # Initialize cluster locally.
        # port = get_open_port()
        # We need to setup the distributed init method to make sure
        # the distributed megatron code (e.g., get world size) works correctly.
        # distributed_init_method = f"tcp://localhost:{port}"
        # Create the LLM engine.
        engine = cls(
            *engine_configs,
            # distributed_init_method,
            # None,
            log_stats=not engine_args.disable_log_stats,
        )
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()
        if prompt_token_ids is None:
            invalidInputError(prompt is not None, "Prompt should not be None")
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params, arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _schedule(
        self,
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, List[RequestOutput]]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        return (
            seq_group_metadata_list,
            scheduler_outputs,
            [
                RequestOutput.from_seq_group(seq_group)
                for seq_group in scheduler_outputs.ignored_seq_groups
            ],
        )

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        invalidInputError(sampling_params.use_beam_search, "Should be beam_search")
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id
        )
        if early_stopping is False:
            highest_attainable_score = best_running_seq.get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id
            )
        else:
            invalidInputError(
                early_stopping == "never", "early_stopping should be never"
            )
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() + sampling_params.max_tokens,
                    self.scheduler_config.max_model_len,
                )
                highest_attainable_score = best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                    seq_len=max_possible_length,
                )
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = best_running_seq.get_beam_search_score(
                    length_penalty=length_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        return current_worst_score >= highest_attainable_score

    def _process_sequence_group_samples(
        self, seq_group: SequenceGroup, samples: List[SequenceOutputs]
    ) -> None:
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {parent_seq.seq_id: [] for parent_seq in parent_seqs}
        # Co(gc):parent_child_dict = {seq_id: [SampleOutputs]}
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        # Co(gc): For each child samples, create a sequence, and add it the child_seqs
        for parent in parent_seqs:
            # Co(gc): Get all the child_samples, SequenceOuptuts
            child_samples: List[SequenceOutputs] = parent_child_dict[parent.seq_id]
            # We do not have any SequenceOutputs
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            # Co(gc): The outputs diverges, we need to fork the requests
            for child_sample in child_samples[:-1]:
                new_child_seq_id = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(
                    child_sample.output_token,
                    child_sample.logprobs,
                    child_sample.latency,
                )
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(
                last_child_sample.output_token,
                last_child_sample.logprobs,
                last_child_sample.latency,
            )
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(seq, seq_group.sampling_params)

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        pass
                        # bigdl-llm change start
                        # summary: fork_seq is doing some block manager ops, so we remove this
                        # self.scheduler.fork_seq(parent, seq)
                        # bigdl-llm change end

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)
            return

        # Beam search case
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = seq_group.sampling_params.best_of
        length_penalty = seq_group.sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False) for seq in existing_finished_seqs]
        new_finished_seqs = [
            (seq, parent, True) for seq, parent in child_seqs if seq.is_finished()
        ]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(
            key=lambda x: x[0].get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id
            ),
            reverse=True,
        )
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        running_child_seqs = [
            (seq, parent) for seq, parent in child_seqs if not seq.is_finished()
        ]
        # Sort the running sequences by their scores.
        running_child_seqs.sort(
            key=lambda x: x[0].get_beam_search_score(
                length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id
            ),
            reverse=True,
        )

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                seq_group.sampling_params.early_stopping,
                seq_group.sampling_params,
                best_running_seq,
                current_worst_seq,
            )

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    pass
                    # bigdl-llm change start
                    # summary: fork_seq is doing some block manager ops, so we remove this
                    # self.scheduler.fork_seq(parent, seq)
                    # bigdl-llm change end

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)

    def _process_model_outputs(
        self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs
    ) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, samples in zip(scheduled_seq_groups, output):
            self._process_sequence_group_samples(seq_group, samples)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in scheduled_seq_groups + scheduler_outputs.ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        # Co(gc): we disable the gpu stats part in the function
        if self.log_stats:
            # Log the system stats.
            self._log_system_stats(
                scheduler_outputs.prompt_run, scheduler_outputs.num_batched_tokens
            )
        return request_outputs

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored

        # Execute the model.
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            finished_seqs=scheduler_outputs.finished_seqs,
        )

        return self._process_model_outputs(output, scheduler_outputs) + ignored

    def _log_system_stats(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.monotonic()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        elapsed_time = now - self.last_logging_time
        if elapsed_time < _LOGGING_INTERVAL_SEC:
            return

        # Discard the old stats.
        self.num_prompt_tokens = [
            (t, n) for t, n in self.num_prompt_tokens if now - t < _LOGGING_INTERVAL_SEC
        ]
        self.num_generation_tokens = [
            (t, n)
            for t, n in self.num_generation_tokens
            if now - t < _LOGGING_INTERVAL_SEC
        ]

        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        # bigdl-llm change start
        # summary: removing logging of pagetable related arguments
        # total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        # num_free_gpu_blocks = (
        #     self.scheduler.block_manager.get_num_free_gpu_blocks())
        # num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        # gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        # total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        # if total_num_cpu_blocks > 0:
        #     num_free_cpu_blocks = (
        #         self.scheduler.block_manager.get_num_free_cpu_blocks())
        #     num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
        #     cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        # else:
        #     cpu_cache_usage = 0.0

        # bigdl-llm change end

        logger.info(
            "Avg prompt throughput: "
            f"{avg_prompt_throughput:.1f} tokens/s, "
            "Avg generation throughput: "
            f"{avg_generation_throughput:.1f} tokens/s, "
            f"Running: {len(self.scheduler.running)} reqs, "
            f"Pending: {len(self.scheduler.waiting)} reqs, "
        )
        self.last_logging_time = now

    def _decode_sequence(self, seq: Sequence, sampling_params: SamplingParams) -> None:
        """Decodes the new token for a sequence."""
        (
            new_tokens,
            new_output_text,
            prefix_offset,
            read_offset,
        ) = detokenize_incrementally(
            self.tokenizer,
            all_input_ids=seq.get_token_ids(),
            prev_tokens=seq.tokens,
            prefix_offset=seq.prefix_offset,
            read_offset=seq.read_offset,
            skip_special_tokens=sampling_params.skip_special_tokens,
        )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence, sampling_params: SamplingParams) -> None:
        """Stop the finished sequences."""
        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                # Truncate the output text so that the stop string is
                # not included in the output.
                seq.output_text = seq.output_text[: -len(stop_str)]
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
        if seq.get_last_token_id() in sampling_params.stop_token_ids:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if (
            not sampling_params.ignore_eos
        ) and seq.get_last_token_id() == self.tokenizer.eos_token_id:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            # bigdl-llm change start
            # summary: we disable ray here
            # if self.parallel_config.worker_use_ray:
            #     executor = partial(worker.execute_method.remote, method)
            # else:
            executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        # if self.parallel_config.worker_use_ray:
        #     all_outputs = ray.get(all_outputs)

        # bigdl-llm change end
        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            invalidInputError(
                output == other_output, "All workers should have same output"
            )
        return output
