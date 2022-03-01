""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, paragraph_plans, paragraph_plans_lengths, paragraph_plans_count, summary_context,
                summary_context_lengths, summary_context_count, tgt, tgt_lengths, bptt=False, tgt_chosen_pp=None,
                training_step=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            :param paragraph_plans: paragraph plans
            :param paragraph_plans_lengths: lengths of paragraph plans
            :param paragraph_plans_count: count of paragraph plans
            :param summary_context: summary context
            :param summary_context_lengths: lengths of segments in summary context
            :param summary_context_count: count of segments in summary context
            :param tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            :param bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        paragraph_plan_selected, chosen_embeddings_init, chosen_pp_lengths, attentions, pp_context_mask, prior_context= self.encoder(
            paragraph_plans, paragraph_plans_lengths, paragraph_plans_count, summary_context, summary_context_lengths,
            summary_context_count, tgt_lengths, tgt_chosen_pp=tgt_chosen_pp, training_step=training_step)
        pp_encoded, pp_sentences, chosen_sentences_id, chosen_sentences_id_with_batch_indices = paragraph_plan_selected
        prior_attentions, posterior_attentions = attentions
        if bptt is False:
            self.decoder.init_state(None, None, chosen_embeddings_init)
        dec_out, attns = self.decoder(tgt, pp_encoded,
                                      memory_lengths=chosen_pp_lengths, prior_context=prior_context)
        return dec_out, attns, prior_attentions, posterior_attentions, pp_context_mask, chosen_sentences_id_with_batch_indices

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
