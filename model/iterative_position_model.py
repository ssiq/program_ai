import tensorflow as tf

from common import rnn_util, rnn_cell, tf_util, code_util

class IterativePositionModel(tf_util.BaseModel):
    def __init__(self,
                 word_embedding_fn,
                 character_embedding_fn,
                 identifier_token,
                 learning_rate,
                 hidden_size,
                 rnn_layer_num):
        super().__init__(learning_rate=learning_rate)
        self._word_embedding_fn = word_embedding_fn
        self._character_embedding_fn = character_embedding_fn
        self._identifier_token = identifier_token
        self._hidden_size = hidden_size
        self._rnn_layer_num = rnn_layer_num

    def _rnn_cell(self):
        return tf.nn.rnn_cell.GRUCell(self._hidden_size)

    def _multi_layer_rnn_cell(self):
        return tf.nn.rnn_cell.MultiRNNCell([self._rnn_cell() for _ in range(self._rnn_layer_num)])

    def _code_embedding(self, word_sequences, character_sequences, character_length):
        return code_util.code_embedding(
            self._word_embedding_fn(word_sequences),
            self._character_embedding_fn(character_sequences, character_length),
            word_sequences,
            self._identifier_token
        )

    def _rnn_seq(self, embedding_seq, token_length):
        return rnn_util.bi_rnn(self._multi_layer_rnn_cell, embedding_seq, token_length)

    def _create_output_with_attention(self, cell, memory, ):
        """
        It is the in-loop function.
        :param cell: a rnn cell object
        :param memory: the input memory
        :return: (output_state, position_logit, is_copy, keyword_logit, copy_word_logit)
        """
        pass

