"""Define RNN-based summary history encoder."""
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory


class HistoryRNNEncoder(EncoderBase):
    """ A rnn encoder for encoding summary history.
    It takes input of representation of summary.
    """

    def __init__(self, rnn_type, num_layers,
                 hidden_size, dropout=0.0):
        super(HistoryRNNEncoder, self).__init__()
        self.rnn, self.no_pack_padded_seq = rnn_factory(rnn_type,
                                                        input_size=hidden_size,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers,
                                                        dropout=dropout,
                                                        bidirectional=False)


    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout)

    def forward(self, emb, enc_state=None, lengths=None, enforce_sorted=None):
        self._check_args(emb, lengths)
        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list, enforce_sorted=enforce_sorted)

        if enc_state is not None:
            memory_bank, encoder_final = self.rnn(packed_emb, enc_state)
        else:
            memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        return encoder_final, memory_bank, lengths
