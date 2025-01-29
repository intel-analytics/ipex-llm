import torch
from torch import nn
from transformers import AutoModel

from context import ContextHead
from moe import ExpertLayer, MoEFFLayer
from transformer import ParallelTransformerAEP, ParallelTransformerBlock


class GRec(nn.Module):
    def __init__(self, deep_dims, page_dim, seq_dim, item_meta_dim, page_embed_dim, seq_embed_dim, item_embed_dim, item_meta_embed_dim, item_pre_embed_dim, deep_embed_dims, wad_embed_dim, nlp_embed_dim, seq_hidden_size, nlp_encoder_path, task_type_dims, task_type_embed_dim, task_out_dims, num_task,
                 num_wide=0, nlp_dim=0, item_freeze=None, item_pre_freeze=None, nlp_freeze=None, context_head_kwargs=None, sequence_transformer_kwargs=None,
                 page_embedding_weight=None, item_embedding_weight=None, item_meta_embedding_weight=None, item_pre_embedding_weight=None, shared_embeddings_weight=None, moe_kwargs=None):
        super().__init__()
        self.nlp_encoder = AutoModel.from_pretrained(nlp_encoder_path)
        sequence_transformer_kwargs = sequence_transformer_kwargs if sequence_transformer_kwargs else {}
        
        if page_embedding_weight is None:
            print("not use pretrained embedding")
            self.page_embedding = nn.Embedding(page_dim, page_embed_dim)
        else:
            print("use pretrained item embedding")
            self.page_embedding = nn.Embedding.from_pretrained(page_embedding_weight, freeze=False)
        if item_embedding_weight is None:
            print("not use pretrained embedding")
            self.item_embedding = nn.Embedding(seq_dim, item_embed_dim)
        else:
            print("use pretrained item embedding")
            self.item_embedding = nn.Embedding.from_pretrained(item_embedding_weight, freeze=False)
        if item_meta_embedding_weight is None:
            print("not use pretrained embedding")
            self.item_meta_embedding = nn.Embedding(item_meta_dim, item_meta_embed_dim)
        else:
            print("use pretrained item embedding")
            self.item_meta_embedding = nn.Embedding.from_pretrained(item_meta_embedding_weight, freeze=False)
        if item_pre_embedding_weight is None:
            print("not use pretrained embedding")
            self.item_pre_embedding = nn.Embedding(seq_dim, item_pre_embed_dim)
        else:
            print("use pretrained item pre embedding")
            self.item_pre_embedding = nn.Embedding.from_pretrained(item_pre_embedding_weight, freeze=False)
            
        if item_freeze:
            self.item_embedding.weight.requires_grad = False
        if item_pre_freeze:
            self.item_pre_embedding.weight.requires_grad = False
            
        if nlp_freeze:
            for param in self.nlp_encoder.parameters():
                param.requires_grad = False
         
        self.combined_dim = nlp_embed_dim + wad_embed_dim + seq_embed_dim + seq_embed_dim + seq_embed_dim
        self.task_embedding = nn.ModuleList([
            nn.Embedding(task_type_dim, task_type_embed_dim)
            for task_type_dim in task_type_dims
        ])
        self.context_head = ContextHead(
            deep_dims=deep_dims,
            num_wide=num_wide,
            item_embedding=self.item_embedding,
            shared_embeddings_weight=shared_embeddings_weight,
            wad_embed_dim=wad_embed_dim,
            deep_embed_dims=deep_embed_dims
        )
        self.sequence_transformer = ParallelTransformerAEP(
            page_embedding=self.page_embedding,
            item_embedding=self.item_embedding,
            item_meta_embedding=self.item_meta_embedding,
            item_pre_embedding=self.item_pre_embedding,
            dim=seq_embed_dim,
            dim_head=seq_embed_dim,
            moe_kwargs=moe_kwargs,
            **sequence_transformer_kwargs
        )
        self.att_pooling = ParallelTransformerBlock(
            dim=256, dim_head=256, heads=1
        )
        self.seq_dense = torch.nn.Linear(
            in_features=seq_embed_dim,
            out_features=seq_embed_dim
        )
        self.nlp_dense = torch.nn.Linear(
            in_features=nlp_dim,
            out_features=nlp_embed_dim
        ) 
        self.moe = MoEFFLayer(dim=seq_embed_dim, num_experts=moe_kwargs.get("num_experts"), expert_capacity=moe_kwargs.get("expert_capacity"), router_jitter_noise=moe_kwargs.get("router_jitter_noise"), hidden_size=seq_embed_dim, expert_class=ExpertLayer)

        self.tasks_dense1 = nn.Linear(
            self.combined_dim, 
            self.combined_dim // 2
        )
        self.tasks_dense2 = nn.Linear(
            self.combined_dim // 2, 
            task_out_dims[0],
            bias=False
        )
        self.tasks_act1 = self.tasks_act2 = nn.LeakyReLU(0.2)
        self.seq_dim = seq_dim
        self.task_type_dim = num_task
    
    def split_task(self, task_type_dim, task_in, combined_out):
        task_indices = []
        task_outs = []
        task_user_outs = []
        for i in range(task_type_dim):
            task_indice = task_in == i
            task_indice = torch.nonzero(task_indice).flatten()
            task_input = combined_out[task_indice]
            task_out = self.tasks_dense1(task_input)
            task_user_out = self.tasks_act1(task_out)
            task_out = self.tasks_dense2(task_user_out)
            task_indices.append(task_indice)
            task_user_outs.append(task_user_out)
            task_outs.append(task_out)
        return task_indices, task_outs, task_user_outs
    
    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
        
    def forward(self, deep_in, page_in, item_in, vl_in, tasks_in, current_in, current_meta_in, item_meta_in=None, page_meta_in=None, item_meta_wide_in=None, page_meta_wide_in=None, wide_in=None, input_ids=None, attention_mask=None, shared_in=None):
        """
        Args:
            deep_in: list, a list of Tensor of shape [batch_size, deep_dims].
            seq_in: Tensor, shape [batch_size, seq_len].
            vl_in: Tensor, shape [batch_size].
            wide_in: list, a list of Tensor of shape [batch_size, num_wide].
            shared_in: list, a list of Tensor of shape [batch_size, num_shared] (default=None).
            search_ids: tensor, Tensor of shape [batch_size, sentence_length] (default=None).
            att_mask: tensor, Tensor of shape [batch_size, sentence_length] (default=None).

        Return:
            out: Tensor, shape [batch_size, seq_dim].
            user_out: Tensor, shape [batch_size, seq_embed_dim].
        """
        search_out = self.nlp_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output.to(dtype=torch.float32)
        search_out = self.nlp_dense(search_out)
        ctx_out = self.context_head(deep_in=deep_in, wide_in=wide_in, shared_in=shared_in)
        seq_out = self.sequence_transformer(page_in=page_in, item_in=item_in, item_meta_in=item_meta_in, page_meta_in=page_meta_in, item_meta_wide_in=item_meta_wide_in, page_meta_wide_in=page_meta_wide_in, vl_in=vl_in)
        seq_out = self.seq_dense(seq_out)
        current_item_out = self.item_embedding(current_in)
        current_meta_out = self.item_meta_embedding(current_meta_in)
        current_pre_out = self.item_pre_embedding(current_in)

        current_out = torch.cat((current_item_out, current_meta_out, current_pre_out), 1)

        tasks_out_list = [self.task_embedding[i](task_in).unsqueeze(1)
                           for i, task_in in enumerate(tasks_in)]
        task_out = torch.cat(tasks_out_list, dim=2).squeeze(1)
        outs = torch.cat((seq_out[:, None, :], ctx_out[:, None, :], search_out[:, None, :], current_out[:, None, :], task_out[:, None, :]), dim=1)
        outs = self.att_pooling(outs)
        outs, aux_loss = self.moe(outs)
        
        outs = outs.reshape(-1, self.combined_dim)
        task_indices, task_outs, task_user_outs = self.split_task(self.task_type_dim, tasks_in[0], outs)
        return (tuple(task_indices), tuple(task_outs), aux_loss)
