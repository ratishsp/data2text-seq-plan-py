import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.summary_history_rnn_encoder import HistoryRNNEncoder
from onmt.modules.global_attention_context import GlobalAttentionContext
from onmt.modules.paragraph_plan_attention import ParagraphPlanSelectionAttention
import random
from collections import Counter


class ParagraphPlanAndSummaryContextEncoder(EncoderBase):
    """ An encoder for classifying the entities/ innings.

    @classmethod
    def from_opt(cls, opt, embeddings):

        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge)
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None, src_summary_ctx_emb=None, gumbel_softmax_temp=None,
                 max_training_steps=100000, min_teacher_forcing_ratio = 0, tgt_lengths_emb=None, tgt_lengths_vocab=None,
                 block_ngram_plan_repeat=0, block_repetitions=0, current_paragraph_index=-1, min_paragraph_count=10,
                 block_consecutive_repetitions=-1, src_vocab=None):
        super(ParagraphPlanAndSummaryContextEncoder, self).__init__()
        assert embeddings is not None
        assert src_summary_ctx_emb is not None
        self.paragraph_plan_encoder = RNNEncoder(rnn_type, bidirectional, num_layers, hidden_size, dropout, embeddings)
        self.summary_ctx_encoder = RNNEncoder(rnn_type, bidirectional, num_layers,
                                              hidden_size, dropout, src_summary_ctx_emb)
        self.attn = GlobalAttentionContext(hidden_size, attn_type="dot")
        self.summary_ctx_attn = GlobalAttentionContext(hidden_size, attn_type="dot")
        self.summary_history_encoder = HistoryRNNEncoder(rnn_type="LSTM", num_layers=num_layers,
                                                         hidden_size=hidden_size, dropout=dropout)
        self.pp_selection_rnn = HistoryRNNEncoder(rnn_type="LSTM", num_layers=num_layers,
                                                  hidden_size=hidden_size, dropout=dropout)
        self.pp_selection_attn = ParagraphPlanSelectionAttention(hidden_size)
        self.gumbel_softmax_temp = gumbel_softmax_temp
        self.hidden_size = hidden_size
        self.prior_query_layer = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.posterior_query_layer = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.prior_context_init = nn.Parameter(torch.Tensor(hidden_size))
        self.pp_state_init = nn.Parameter(torch.Tensor(hidden_size))
        self.max_training_steps = max_training_steps
        self.min_teacher_forcing_ratio = min_teacher_forcing_ratio
        self.block_ngram_plan_repeat = block_ngram_plan_repeat
        self.block_repetitions = block_repetitions
        self.tgt_lengths_emb = tgt_lengths_emb
        self.tgt_lengths_vocab = tgt_lengths_vocab
        self.current_paragraph_index = current_paragraph_index
        self.min_paragraph_count = min_paragraph_count
        self.block_consecutive_repetitions = block_consecutive_repetitions
        self.src_vocab = src_vocab

    @classmethod
    def from_opt(cls, opt, embeddings=None, src_summary_ctx_emb=None, tgt_lengths_emb=None, tgt_lengths_vocab=None,
                 block_ngram_plan_repeat=None, block_repetitions=None, current_paragraph_index=None, min_paragraph_count=None,
                 block_consecutive_repetitions=-1, src_vocab=None):
        return cls(
            opt.rnn_type,
            True,  # for bidirectional
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            src_summary_ctx_emb,
            opt.gumbel_softmax_temp,
            opt.max_training_steps,
            opt.min_teacher_forcing_ratio,
            tgt_lengths_emb,
            tgt_lengths_vocab,
            block_ngram_plan_repeat,
            block_repetitions,
            current_paragraph_index,
            min_paragraph_count,
            block_consecutive_repetitions,
            src_vocab)

    def forward(self, paragraph_plans, paragraph_plans_lengths=None, paragraph_plans_count=None, summary_context=None,
                summary_context_lengths=None, summary_context_count=None, tgt_lengths=None, training=True, tgt_chosen_pp=None,
                training_step=None, input_pp=None):
        self._check_args(paragraph_plans, paragraph_plans_lengths)
        # Encode paragraph plans
        paragraph_plans_lengths_non_zero = paragraph_plans_lengths != -1
        paragraph_plans_non_zero = paragraph_plans[:, paragraph_plans_lengths_non_zero]
        paragraph_plans_lengths_non_zero = paragraph_plans_lengths[paragraph_plans_lengths != -1]  # exclude src length padding tokens
        encoder_final, memory_bank, paragraph_lengths = self.paragraph_plan_encoder(paragraph_plans_non_zero,
                                                                                    paragraph_plans_lengths_non_zero,
                                                                                    enforce_sorted=False)
        pp_context = self.attn(memory_bank.transpose(0, 1), memory_lengths=paragraph_plans_lengths_non_zero)
        segment_length, _, _ = memory_bank.size()
        memory_bank_ = memory_bank.transpose(0, 1)
        segment_list = memory_bank_.split(paragraph_plans_count.tolist(), dim=0)
        memory_bank = pad_sequence(segment_list)  # segment_count, batch, segment_len, hidden_size
        memory_bank = memory_bank.transpose(0, 1) # batch, segment_count, segment_len, hidden_size
        pp_context_ = pp_context.split(paragraph_plans_count.tolist(), dim=0)
        pp_context = pad_sequence(pp_context_).transpose(0, 1)
        batch_ = paragraph_plans_count.size()[0]
        encoder_final = tuple(
            self._pad_encoder_final(enc_hid, paragraph_plans_count) for enc_hid in encoder_final) # (batch, segment_count, 2, hidden_size/2)
        # sequential paragraph plan selection
        assert paragraph_plans_lengths.size()[0] % batch_ == 0
        segment_count = paragraph_plans_lengths.size()[0] // batch_

        prior_context = self.prior_context_init.unsqueeze(0).repeat(batch_, 1).unsqueeze(1)
        pp_length_ = 1
        posterior_context = None
        paragraph_batch_ = batch_
        if summary_context is not None:
            self._check_args(summary_context, summary_context_lengths)
            # Encode summary context for posterior
            summary_encoder_final, summary_memory_bank, summary_lengths = self.summary_ctx_encoder(summary_context,
                                                                                                   summary_context_lengths,
                                                                                                   enforce_sorted=False)
            paragraph_batch_ = summary_memory_bank.size()[1]
            assert summary_lengths.size() == summary_context_lengths.size()
            summary_weighted_sum_context = self.summary_ctx_attn(summary_memory_bank.transpose(0, 1),
                                                                 memory_lengths=summary_lengths)
            assert batch_ == summary_context_count.size()[0]
            assert summary_weighted_sum_context.size()[0] == paragraph_batch_
            assert paragraph_batch_ % batch_ == 0
            pp_length_ = paragraph_batch_ // batch_
            summary_weighted_sum_context = summary_weighted_sum_context.view(batch_, pp_length_,
                                                                             self.hidden_size).transpose(0, 1)
            # assert summary_context_count.sum() == summary_weighted_sum_context.size()[0]
            summary_history_encoder_final, summary_history_memory_bank, summary_history_lengths = self.summary_history_encoder(
                summary_weighted_sum_context, lengths=summary_context_count, enforce_sorted=False)
            assert summary_history_memory_bank.size()[0] == pp_length_
            assert summary_history_memory_bank.size()[1] == batch_
            posterior_context = summary_history_memory_bank.transpose(0, 1)

            # Encode summary context for prior
            if training:
                prior_context = torch.cat((prior_context, posterior_context[:, :-1, :]), dim=1)
            else:
                prior_context = torch.cat((prior_context, posterior_context), dim=1)
                pp_length_ += 1
                paragraph_batch_ = pp_length_ * batch_

        prior, posterior, pp_states = self.sequential_pp_selection(batch_, paragraph_plans_count,
                                                                   posterior_context, pp_context, pp_length_,
                                                                   prior_context, training, input_pp)
        prior_attentions, prior_argmaxes = prior
        posterior_attentions, posterior_argmaxes = posterior
        use_teacher_forcing = False
        if training and tgt_chosen_pp is not None:
            teacher_forcing_ratio = max(1 - training_step / self.max_training_steps, self.min_teacher_forcing_ratio)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
        # Decoding
        if training and use_teacher_forcing and tgt_chosen_pp is not None:
            chosen_sentences_id_with_batch = tgt_chosen_pp.transpose(0, 1).view(batch_, -1)
            chosen_sentences_id_with_batch = torch.nn.functional.one_hot(chosen_sentences_id_with_batch, segment_count).float()
        elif training:
            chosen_sentences_id_with_batch = torch.cat(posterior_argmaxes, dim=1)
        else:
            chosen_sentences_id_with_batch = self.block_ngram_repeats(posterior, prior)
            argmaxes = chosen_sentences_id_with_batch.argmax(dim=-1)[:, -1]
            for _batch_index, prior_value in enumerate(argmaxes):
                if prior_value == 4:
                    tgt_lengths[_batch_index, -1, 0] = self.tgt_lengths_vocab.stoi['<end-summary-bin>']

        paragraph_plans_lengths = paragraph_plans_lengths.view(batch_, -1).unsqueeze(-1)
        chosen_pp_lengths = torch.bmm(chosen_sentences_id_with_batch, paragraph_plans_lengths.float())
        chosen_pp_lengths = chosen_pp_lengths.view(-1).long()
        encoder_final_size = encoder_final[0].size()

        chosen_sentences_id_with_batch_indices = chosen_sentences_id_with_batch.argmax(dim=-1)
        chosen_sentences_id_with_batch = chosen_sentences_id_with_batch.unsqueeze(2)
        chosen_embeddings_init = tuple(enc_hid.view(batch_, segment_count, -1).unsqueeze(1) for enc_hid in encoder_final)
        chosen_embeddings_init = tuple(torch.matmul(chosen_sentences_id_with_batch, enc_hid) for enc_hid in chosen_embeddings_init)
        chosen_embeddings_init = tuple(enc_hid.view(paragraph_batch_, encoder_final_size[2], self.hidden_size//2) for enc_hid in chosen_embeddings_init)
        chosen_embeddings_init = tuple(enc_hid.transpose(0, 1) for enc_hid in chosen_embeddings_init)

        paragraph_plans = paragraph_plans.transpose(0, 1).view(batch_, segment_count, segment_length).unsqueeze(1)
        chosen_sentences = torch.matmul(chosen_sentences_id_with_batch, paragraph_plans.float())
        chosen_sentences = chosen_sentences.view(batch_, pp_length_, segment_length).long()
        if not training:
            argmaxes = chosen_sentences_id_with_batch.argmax(dim=-1)[:, -1]
            for _batch_index, prior_value in enumerate(argmaxes):
                chosen_sentence = chosen_sentences[_batch_index, -1, :].tolist()
                if self.src_vocab.stoi['<end-plan>'] not in chosen_sentence and self.src_vocab.stoi[
                    '<INNING>'] not in chosen_sentence:  # if selected pp is an entity, have a shorter bucket
                    tgt_lengths[_batch_index, -1, 0] = self.tgt_lengths_vocab.stoi['<LENGTH0>']
        tgt_lengths_emb_values = self.tgt_lengths_emb(tgt_lengths)
        memory_bank = memory_bank.view(batch_, segment_count, segment_length * self.hidden_size)#.unsqueeze(1)
        chosen_embeddings = torch.einsum('abij,ajk->abik', chosen_sentences_id_with_batch, memory_bank)
        chosen_embeddings = chosen_embeddings.squeeze(1).view(-1, segment_length, self.hidden_size).transpose(0, 1)
        chosen_embeddings = torch.cat([tgt_lengths_emb_values.view(-1, self.hidden_size).unsqueeze(0), chosen_embeddings], dim=0)
        chosen_pp_lengths = chosen_pp_lengths + 1
        prior_context = prior_context.view(paragraph_batch_, self.hidden_size)
        current_chosen_sentences_id = chosen_sentences_id_with_batch.squeeze(2)[:, -1, :].argmax(dim=1)
        return (chosen_embeddings, chosen_sentences, current_chosen_sentences_id, chosen_sentences_id_with_batch_indices), chosen_embeddings_init, chosen_pp_lengths, (prior_attentions, posterior_attentions), paragraph_plans_count, prior_context

    def block_ngram_repeats(self, posterior, prior):
        prior_attentions, prior_argmaxes = prior
        chosen_sentences_id_with_batch = torch.cat(prior_argmaxes, dim=1)  # (batch, steps)
        posterior_attentions, posterior_argmaxes = posterior
        if posterior_argmaxes:
            posterior_argmaxes_cat = torch.cat(posterior_argmaxes, dim=1)
            cur_len = posterior_attentions.size()[1]
            updated_predictions = []
            if self.block_ngram_plan_repeat > 0 and prior_attentions.size()[1] > self.block_ngram_plan_repeat:
                for batch_id in range(prior_attentions.size()[0]):
                    hyp = posterior_argmaxes_cat[batch_id].argmax(dim=-1)
                    ngrams = set()
                    gram = []
                    for i in range(cur_len):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram + [hyp[i].item()])[-self.block_ngram_plan_repeat:]
                        ngrams.add(tuple(gram))
                    last_n_minus_1_gram = hyp[-self.block_ngram_plan_repeat+1:].tolist()
                    top_current_predictions = prior_attentions[batch_id, -1].topk(k=50, dim=-1)[1].tolist()  # get the top 50 predictions
                    for prediction in top_current_predictions:
                        if tuple(last_n_minus_1_gram + [prediction]) in ngrams:
                            continue
                        if self.block_repetitions > 0 and Counter(hyp.tolist())[prediction] >= self.block_repetitions - 1:  # block repetitions
                            continue
                        if 0 < self.current_paragraph_index < self.min_paragraph_count and prediction == 4:  # skip end-plan
                            # if the current paragraph index is smaller than the total paragraph count
                            continue
                        if self.block_consecutive_repetitions > -1 and prediction == hyp[-1].item():
                            continue
                        updated_predictions.append(prediction)
                        break
                pp_argmax = nn.functional.one_hot(torch.tensor(updated_predictions, device=prior_attentions.device),
                                                         num_classes=prior_attentions.size(2)).float()
                chosen_sentences_id_with_batch[:, -1, :] = pp_argmax
        return chosen_sentences_id_with_batch

    def sequential_pp_selection(self, batch_, paragraph_plans_count,
                                posterior_context, pp_context, pp_length_,
                                prior_context, training, input_pp):
        pp_states = []
        prior_argmaxes = []
        posterior_argmaxes = []
        pp_state = self.pp_state_init.unsqueeze(0).repeat(batch_, 1)
        prior_attentions = []
        posterior_attentions = []
        enc_state = None
        if input_pp is not None:
            input_pp_tensor = torch.tensor(input_pp, device=prior_context.device).long()
        for current_pp in range(pp_length_):
            current_prior_context = prior_context[:, current_pp, :]
            current_prior_context = current_prior_context.view(batch_, self.hidden_size)
            # Make query
            current_prior_query = self.prior_query_layer(torch.cat([current_prior_context, pp_state], dim=1))
            # Compute attention
            prior_pp_attention, prior_pp_argmax = self.compute_pp_attention(pp_context,
                                                                            current_prior_query,
                                                                            paragraph_plans_count,
                                                                            use_gumbel=False)
            prior_attentions.append(prior_pp_attention)
            prior_argmaxes.append(prior_pp_argmax)
            if posterior_context is not None and current_pp < posterior_context.size(1):
                current_posterior_context = posterior_context[:, current_pp, :]
                current_posterior_context = current_posterior_context.view(batch_, self.hidden_size)
                current_posterior_query = self.posterior_query_layer(
                    torch.cat([current_posterior_context, pp_state], dim=1))
                posterior_pp_attention, posterior_pp_argmax = self.compute_pp_attention(pp_context,
                                                                                                      current_posterior_query,
                                                                                                      paragraph_plans_count,
                                                                                                      use_gumbel=training)
                if not training:
                    pp_argmax = input_pp_tensor[:, current_pp: current_pp + 1]
                    posterior_pp_argmax = nn.functional.one_hot(pp_argmax,
                                                                       num_classes=posterior_pp_attention.size(
                                                                           2)).float()
                # Sample pp from posterior
                chosen_pps = torch.bmm(posterior_pp_argmax, pp_context)
                # Roll out one step
                chosen_pps = chosen_pps.transpose(0, 1)  # 1, batch_, dim
                lengths = torch.ones(batch_, device=prior_context.device).long()
                enc_state, pp_state, _ = self.pp_selection_rnn(chosen_pps, enc_state=enc_state,
                                                                      lengths=lengths)
                pp_state = pp_state.view(batch_, self.hidden_size)
                pp_states.append(pp_state)
                posterior_attentions.append(posterior_pp_attention)
                posterior_argmaxes.append(posterior_pp_argmax)
        prior_attentions = torch.cat(prior_attentions, dim=1)
        if posterior_context is not None:
            posterior_attentions = torch.cat(posterior_attentions, dim=1)
        return (prior_attentions, prior_argmaxes), (posterior_attentions, posterior_argmaxes), pp_states

    def compute_pp_attention(self, pp_context, current_query, pp_context_mask, use_gumbel=False):
        pp_attention = self.pp_selection_attn(current_query.unsqueeze(1), pp_context,
                                                             memory_lengths=pp_context_mask)
        if self.gumbel_softmax_temp > 0 and use_gumbel:
            pp_argmax = nn.functional.gumbel_softmax(pp_attention, tau=self.gumbel_softmax_temp,
                                                            hard=True, dim=-1)
        else:
            pp_argmax = torch.argmax(pp_attention, dim=-1)
            pp_argmax = nn.functional.one_hot(pp_argmax, num_classes=pp_attention.size(2)).float()
        return pp_attention, pp_argmax

    def _pad_encoder_final(self, enc_hidden, paragraph_plans_count):
        enc_hidden_ = enc_hidden.transpose(0, 1).split(paragraph_plans_count.tolist(), dim=0)
        enc_hidden_ = pad_sequence(enc_hidden_)
        enc_hidden_ = enc_hidden_.transpose(0, 1)  # batch, segment_count, 2, hidden_size//2
        return enc_hidden_
