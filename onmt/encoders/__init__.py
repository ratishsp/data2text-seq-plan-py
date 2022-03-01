"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder
from onmt.encoders.paragraph_plan_selection_encoder import ParagraphPlanAndSummaryContextEncoder
from onmt.encoders.summary_history_rnn_encoder import HistoryRNNEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder, "classifier": ParagraphPlanAndSummaryContextEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder", "ParagraphPlanAndSummaryContextEncoder",
           "HistoryRNNEncoder", "MeanEncoder", "str2enc"]
