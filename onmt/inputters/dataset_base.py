# coding: utf-8

from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from torchtext.vocab import Vocab


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _add_tgt_paragraph_plan(example, tgt_chosen_pp_field, tgt_field_attrib_name):
    """
    Add field for int version of tgt_plan
    :param example:
    :param tgt_chosen_pp_field:
    :param tgt_field_attrib_name:
    :return:
    """
    tgt_plan = tgt_chosen_pp_field.tokenize(example["tgt_chosen_pp"])
    example[tgt_field_attrib_name] = torch.LongTensor([int(w) for w in tgt_plan])


def _add_ctx_segment_details(example, src_field, src_field_attrib_name, length_attrib, count_attrib, prefix=False):
    """
    Creates fields for total number of records for each segment and total number of segment

    :param example:
    :param src_field:
    :param src_field_attrib_name:
    :param length_attrib:
    :param count_attrib:
    :return:
    """
    _prefix_code = [1, 1, 1, 1, 1, 1] if prefix else []  # UNK, PAD, BOS, EOS, <end-plan> <end-segment>
    indices, segment_lengths, src = set_segment_lengths(_prefix_code, example, src_field, src_field_attrib_name)
    example[length_attrib] = torch.LongTensor(segment_lengths)
    example[count_attrib] = torch.LongTensor([len(segment_lengths)])
    assert len(segment_lengths) == len(indices) + len(_prefix_code)
    assert sum(segment_lengths) + len(segment_lengths) - len(_prefix_code) == len(src)


def set_segment_lengths(_prefix_code, example, src_field, src_field_attrib_name):
    src = src_field.tokenize(example[src_field_attrib_name])
    indices = [index for index, x in enumerate(src) if x == "<segment>"]
    segment_lengths = _prefix_code + [t - s - 1 for s, t in
                                      zip(indices, indices[1:])]  # -1 discounting for <segment> marker
    segment_lengths = segment_lengths + [(len(src) - indices[-1]) - 1]  # -1 discounting for <segment> marker
    return indices, segment_lengths, src


def _add_tgt_lengths(example, field, field_attrib_name, length_attrib):
    _prefix_code = []
    indices, segment_lengths, src = set_segment_lengths(_prefix_code, example, field, field_attrib_name)
    example[length_attrib] = get_length_bin(segment_lengths)
    assert len(segment_lengths) == len(indices) + len(_prefix_code)  # +4 for UNK, PAD, BOS, EOS
    assert sum(segment_lengths) + len(segment_lengths) - len(_prefix_code) == len(src)

"""
(25, 7783) 113240
(40, 7362) 119887
counts 112802
"""
def get_length_bin(segment_lengths):
    output_lengths = []
    for length in segment_lengths:
        if length == 1:
            length_bin = "<end-summary-bin>"
        elif length <= 25:
            length_bin = "<LENGTH0>"
        elif length <= 40:
            length_bin = "<LENGTH1>"
        else:
            length_bin = "<LENGTH2>"
        output_lengths.append(length_bin)

    return output_lengths


def _dynamic_dict(example, src_field, tgt_field, inference=False):
    """Create copy-vocab and numericalize with it.

    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src = src_field.tokenize(example["src"])
    # make a small vocab containing just the tokens in the source sequence
    unk = src_field.unk_token
    pad = src_field.pad_token
    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
    example["src_map"] = src_map
    example["src_ex_vocab"] = src_ex_vocab

    if "tgt" in example and not inference:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask
    return src_ex_vocab, example


class Dataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(self, fields, readers, data, dirs, sort_key,
                 filter_pred=None, inference=False):
        self.sort_key = sort_key
        can_copy = 'src_map' in fields and 'alignment' in fields

        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                      in zip(readers, data, dirs)]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            if can_copy:
                src_field = fields['src']
                tgt_field = fields['tgt']
                # this assumes src_field and tgt_field are both text
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_field.base_field, tgt_field.base_field, inference=inference)
                self.src_vocabs.append(src_ex_vocab)
            _add_ctx_segment_details(ex_dict, fields['src'].base_field, "src", "pp_lengths", "pp_count",
                                 prefix=True)
            if inference and "tgt" not in ex_dict:
                ex_dict["tgt_lengths"] = ["<LENGTH0>"]
            if "tgt" in ex_dict:
                ex_dict['src_summary_ctx'] = ex_dict['tgt']
                _add_ctx_segment_details(ex_dict, fields['src_summary_ctx'].base_field, "src_summary_ctx",
                                     "summary_context_lengths", "summary_context_count", prefix=False)
                if inference:
                    ex_dict["tgt_lengths"] = ["<LENGTH0>"] + ex_dict['tgt'].count("<segment>") * ["<LENGTH2>"]
                else:
                    _add_tgt_lengths(ex_dict, fields['tgt'].base_field, "tgt", "tgt_lengths")

            if "tgt_chosen_pp" in ex_dict:
                _add_tgt_paragraph_plan(ex_dict, fields['tgt'].base_field, "tgt_chosen_pp")
            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)
