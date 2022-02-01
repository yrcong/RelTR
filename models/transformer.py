# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
"""
RelTR Transformer class.
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, )

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, entity_embed, triplet_embed, pos_embed, so_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        entity_embed, entity = torch.split(entity_embed, c, dim=1)
        triplet_embed, triplet = torch.split(triplet_embed, [c, 2 * c], dim=1)

        entity_embed = entity_embed.unsqueeze(1).repeat(1, bs, 1)
        triplet_embed = triplet_embed.unsqueeze(1).repeat(1, bs, 1)
        entity = entity.unsqueeze(1).repeat(1, bs, 1)
        triplet = triplet.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, hs_t, sub_maps, obj_maps = self.decoder(entity, triplet, memory, memory_key_padding_mask=mask,
                                                    pos=pos_embed, entity_pos=entity_embed,
                                                    triplet_pos=triplet_embed, so_pos=so_embed)

        so_masks = torch.cat((sub_maps.reshape(sub_maps.shape[0], bs, sub_maps.shape[2], 1, h, w),
                              obj_maps.reshape(obj_maps.shape[0], bs, obj_maps.shape[2], 1, h, w)), dim=3)

        return hs.transpose(1, 2), hs_t.transpose(1, 2), so_masks, memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, entity, triplet, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, entity_pos: Optional[Tensor] = None,
                triplet_pos: Optional[Tensor] = None, so_pos: Optional[Tensor] = None):
        output_entity = entity
        output_triplet = triplet
        intermediate_entity = []
        intermediate_triplet = []
        intermediate_submaps = []
        intermediate_objmaps = []

        for layer in self.layers:
            output_entity, output_triplet, sub_maps, obj_maps = layer(output_entity, output_triplet, entity_pos, triplet_pos, so_pos,
                                                                      memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                                                      memory_key_padding_mask=memory_key_padding_mask, pos=pos)


            if self.return_intermediate:
                intermediate_entity.append(output_entity)
                intermediate_triplet.append(output_triplet)
                intermediate_submaps.append(sub_maps)
                intermediate_objmaps.append(obj_maps)

        if self.return_intermediate:
            return torch.stack(intermediate_entity), torch.stack(intermediate_triplet), \
                   torch.stack(intermediate_submaps), torch.stack(intermediate_objmaps)


class TransformerDecoderLayer(nn.Module):
    """triplet decoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.activation = _get_activation_fn(activation)

        # entity part
        self.self_attn_entity = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2_entity = nn.Dropout(dropout)
        self.norm2_entity = nn.LayerNorm(d_model)

        self.cross_attn_entity = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1_entity = nn.Dropout(dropout)
        self.norm1_entity = nn.LayerNorm(d_model)

        # triplet part
        self.self_attn_so = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2_so = nn.Dropout(dropout)
        self.norm2_so = nn.LayerNorm(d_model)

        self.cross_attn_sub = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1_sub = nn.Dropout(dropout)
        self.norm1_sub = nn.LayerNorm(d_model)
        self.cross_sub_entity = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2_sub = nn.Dropout(dropout)
        self.norm2_sub = nn.LayerNorm(d_model)

        self.cross_attn_obj = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1_obj = nn.Dropout(dropout)
        self.norm1_obj = nn.LayerNorm(d_model)
        self.cross_obj_entity = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2_obj = nn.Dropout(dropout)
        self.norm2_obj = nn.LayerNorm(d_model)

        # ffn
        self.linear1_entity = nn.Linear(d_model, dim_feedforward)
        self.dropout3_entity = nn.Dropout(dropout)
        self.linear2_entity = nn.Linear(dim_feedforward, d_model)
        self.dropout4_entity = nn.Dropout(dropout)
        self.norm3_entity = nn.LayerNorm(d_model)

        self.linear1_sub = nn.Linear(d_model, dim_feedforward)
        self.dropout3_sub = nn.Dropout(dropout)
        self.linear2_sub = nn.Linear(dim_feedforward, d_model)
        self.dropout4_sub = nn.Dropout(dropout)
        self.norm3_sub = nn.LayerNorm(d_model)

        self.linear1_obj = nn.Linear(d_model, dim_feedforward)
        self.dropout3_obj = nn.Dropout(dropout)
        self.linear2_obj = nn.Linear(dim_feedforward, d_model)
        self.dropout4_obj = nn.Dropout(dropout)
        self.norm3_obj = nn.LayerNorm(d_model)

    def forward_ffn_entity(self, tgt):
        tgt2 = self.linear2_entity(self.dropout3_entity(self.activation(self.linear1_entity(tgt))))
        tgt = tgt + self.dropout4_entity(tgt2)
        tgt = self.norm3_entity(tgt)
        return tgt
    def forward_ffn_sub(self, tgt):
        tgt2 = self.linear2_sub(self.dropout3_sub(self.activation(self.linear1_sub(tgt))))
        tgt = tgt + self.dropout4_sub(tgt2)
        tgt = self.norm3_sub(tgt)
        return tgt
    def forward_ffn_obj(self, tgt):
        tgt2 = self.linear2_obj(self.dropout3_obj(self.activation(self.linear1_obj(tgt))))
        tgt = tgt + self.dropout4_obj(tgt2)
        tgt = self.norm3_obj(tgt)
        return tgt

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt_entity, tgt_triplet, entity_pos, triplet_pos, so_pos,
                memory, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        # entity layer
        q_entity = k_entity = self.with_pos_embed(tgt_entity, entity_pos)
        tgt2_entity = self.self_attn_entity(q_entity, k_entity, value=tgt_entity, attn_mask=tgt_mask,
                                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt_entity = tgt_entity + self.dropout2_entity(tgt2_entity)
        tgt_entity = self.norm2_entity(tgt_entity)

        tgt2_entity = self.cross_attn_entity(query=self.with_pos_embed(tgt_entity, entity_pos),
                                             key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask,
                                             key_padding_mask=memory_key_padding_mask)[0]
        tgt_entity = tgt_entity + self.dropout1_entity(tgt2_entity)
        tgt_entity = self.norm1_entity(tgt_entity)
        tgt_entity = self.forward_ffn_entity(tgt_entity)

        # triplet layer
        # coupled self attention
        t_num = triplet_pos.shape[0]
        h_dim = triplet_pos.shape[2]
        tgt_sub, tgt_obj = torch.split(tgt_triplet, h_dim, dim=-1)
        q_sub = k_sub = self.with_pos_embed(self.with_pos_embed(tgt_sub, triplet_pos), so_pos[0])
        q_obj = k_obj = self.with_pos_embed(self.with_pos_embed(tgt_obj, triplet_pos), so_pos[1])
        q_so = torch.cat((q_sub, q_obj), dim=0)
        k_so = torch.cat((k_sub, k_obj), dim=0)
        tgt_so = torch.cat((tgt_sub, tgt_obj), dim=0)

        tgt2_so = self.self_attn_so(q_so, k_so, tgt_so)[0]
        tgt_so = tgt_so + self.dropout2_so(tgt2_so)
        tgt_so = self.norm2_so(tgt_so)
        tgt_sub, tgt_obj = torch.split(tgt_so, t_num, dim=0)

        # subject branch - decoupled visual attention
        tgt2_sub, sub_maps = self.cross_attn_sub(query=self.with_pos_embed(tgt_sub, triplet_pos),
                                                 key=self.with_pos_embed(memory, pos),
                                                 value=memory, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)
        tgt_sub = tgt_sub + self.dropout1_sub(tgt2_sub)
        tgt_sub = self.norm1_sub(tgt_sub)

        # subject branch - decoupled entity attention
        tgt2_sub = self.cross_sub_entity(query=self.with_pos_embed(tgt_sub, triplet_pos),
                                         key=tgt_entity, value=tgt_entity)[0]
        tgt_sub = tgt_sub + self.dropout2_sub(tgt2_sub)
        tgt_sub = self.norm2_sub(tgt_sub)
        tgt_sub = self.forward_ffn_sub(tgt_sub)

        # object branch - decoupled visual attention
        tgt2_obj, obj_maps = self.cross_attn_obj(query=self.with_pos_embed(tgt_obj, triplet_pos),
                                                 key=self.with_pos_embed(memory, pos),
                                                 value=memory, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)
        tgt_obj = tgt_obj + self.dropout1_obj(tgt2_obj)
        tgt_obj = self.norm1_obj(tgt_obj)

        # object branch - decoupled entity attention
        tgt2_obj = self.cross_obj_entity(query=self.with_pos_embed(tgt_obj, triplet_pos),
                                         key=tgt_entity, value=tgt_entity)[0]
        tgt_obj = tgt_obj + self.dropout2_obj(tgt2_obj)
        tgt_obj = self.norm2_obj(tgt_obj)
        tgt_obj = self.forward_ffn_obj(tgt_obj)

        tgt_triplet = torch.cat((tgt_sub, tgt_obj), dim=-1)
        return tgt_entity, tgt_triplet, sub_maps, obj_maps


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
