"""Module for prediction of chosen paragraph plan"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import aeq, sequence_mask


class ParagraphPlanSelectionAttention(nn.Module):

    def __init__(self, dim):
        super(ParagraphPlanSelectionAttention, self).__init__()
        self.dim = dim

    def score(self, h_t, h_s):
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(src_dim, self.dim)
        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) * (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def forward(self, source, memory_bank, memory_lengths=None):
        """
        Args
        :param source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
        :param memory_bank (FloatTensor): source vectors ``(batch, src_len, dim)``
        :param memory_lengths (LongTensor): the source context lengths ``(batch,)``
        :return:
        """
        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # make it broadcastable
            align.masked_fill_(~mask, -float('inf'))

        align_vectors = F.log_softmax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        return align_vectors
