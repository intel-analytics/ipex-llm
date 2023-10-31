from typing import Dict, List, Optional, Tuple

import torch
from xformers.ops import AttentionBias

from vllm.sequence import SequenceData
from bigdl.llm.vllm.structure.sampling_params import SamplingParams


class InputMetadata:
    """Metadata for input sequences. Used for PagedAttention.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        context_lens: the length of attention context for each generation token.
        max_context_len: The maximum context length.
        block_tables: The block tables. (Seq id -> list of physical block)
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_data: Dict[int, SequenceData],
        prompt_lens: List[int],
        context_lens: torch.Tensor,
        max_context_len: int,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.context_lens = context_lens
        self.max_context_len = max_context_len

        self.to_cache = None

        self.num_prompts = len(prompt_lens)
        self.num_prompt_tokens = sum(prompt_lens)
        self.num_generation_tokens = context_lens.shape[0]

        # Set during the execution of the first attention op.
        self.attn_bias: List[AttentionBias] = []

    def __repr__(self) -> str:
        # Print only useful metadata.
        return (f'InputMetadata('
                f'num_valid_tokens={self.num_valid_tokens}, '
                f'num_prompt_tokens={self.num_prompt_tokens}, '
                f'num_prompts={self.num_prompts}, '
                f'prompt_lens={self.prompt_lens}, '
                f'num_generation_tokens={self.num_generation_tokens}, '
                f'context_lens={self.context_lens}, '
                f'max_context_len={self.max_context_len}), '
                f'max_num_blocks_per_seq={self.max_num_blocks_per_seq}, '
                f'block_tables={self.block_tables}), '
                f'slot_mapping={self.slot_mapping}')
