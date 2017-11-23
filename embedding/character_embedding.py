import abc

from common import util, tf_util
import functools
import more_itertools
import itertools
import tensorflow as tf
import numpy as np

class CharacterEmbedding(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, token_set: set, n_gram=1, embedding_shape=300):
        """
        :param token_set: a set of all characters
        """
        self.BEGIN = "<BEGIN>"
        self.END = "<END>"
        self.preprocess_token = lambda x: more_itertools.windowed([self.BEGIN] + list(x) + [self.END], n_gram)
        self.preprocess_token_without_label = lambda x: more_itertools.windowed(list(x), n_gram)
        token_set = set(more_itertools.flatten(map(lambda x: list(self.preprocess_token(x)) , token_set)))
        self.id_to_character_dict = dict(list(enumerate(start=0, iterable=token_set)))
        self.character_to_id_dict = util.reverse_dict(self.id_to_character_dict)
        self.embedding_shape = embedding_shape

    def _create_embedding_matrix(self):
        with tf.variable_scope("character_embedding_op"):
            m = tf.Variable(np.random.randn(len(self.id_to_character_dict), self.embedding_shape),
                            name="embedding",
                            dtype=tf.float32)
            print("character_embedding_matrix:{}".format(m))
            return m

    def parse_string(self, string_list):
        """
        :param string_list: a list of list of tokens
        :return: a list of list of list of characters of tokens
        """
        max_string_len = max(map(lambda x: len(x)+2, more_itertools.collapse(string_list)))
        max_text_len = max(map(lambda x:len(x), string_list))
        print("max string len:{}".format(max_string_len))
        print("max text len:{}".format(max_text_len))
        def parse_toke(token):
            token = self.preprocess_token(token)
            token = [self.character_to_id_dict[c] for c in token]
            token = token+[0]*(max_string_len-len(token))
            return token

        string_list = [[parse_toke(t) for t in l] for l in string_list]
        empty_token = [0]*max_string_len
        string_list = [l+list(itertools.repeat(empty_token, times=max_text_len-len(l))) for l in string_list]
        return string_list

    def parse_token(self, token, character_position_label=True):
        if character_position_label:
            token = self.preprocess_token(token)
        else:
            token = self.preprocess_token_without_label(token)
        token = [self.character_to_id_dict[c] for c in token]
        return token

    def parse_string_without_padding(self, string_list, token_position_label=True, character_position_label=True):
        '''
        parse string list to a char list
        :param string_list: a list of list of tokens
        :return: a list of list of list of characters of tokens
        '''
        # string_list = [[parse_token(token) for token in l] for l in string_list]
        new_string_list = []
        for l in string_list:
            token_list = []
            try:
                for token in l:
                    token_list.append(self.parse_token(token, character_position_label))
            except Exception as e:
                token_list = None
            new_string_list.append(token_list)
        for s in new_string_list:
            if s == None:
                continue
            if token_position_label:
                s.insert(0, [self.character_to_id_dict[tuple([self.BEGIN])], self.character_to_id_dict[tuple([self.END])]])
                s.append([self.character_to_id_dict[tuple([self.BEGIN])], self.character_to_id_dict[tuple([self.END])]])
        return new_string_list

    @abc.abstractmethod
    def create_embedding_layer(self):
        """
        :return: a function used to create the embedding layer
        """
        pass



class AbstractRNNCharacterEmbedding(CharacterEmbedding):
    def __init__(self, token_set: set, n_gram,  embedding_shape=300):
        super().__init__(token_set, n_gram=n_gram,  embedding_shape=embedding_shape)

    @abc.abstractmethod
    def _rnn_cell(self) -> tf.nn.rnn_cell.BasicRNNCell:
        pass

    @abc.abstractmethod
    def _rnn(self, input_op, input_length):
        pass

    def create_embedding_layer(self):
        def embedding_layer(input_op, input_length_op):
            input_shape = tf_util.get_shape(input_op)
            input_op = tf.reshape(input_op, (-1, input_shape[-1]))
            # length = tf_util.length(tf.one_hot(input_op, len(self.id_to_character_dict)))
            length = tf.reshape(input_length_op, (-1, ))
            # length = tf.where(tf.equal(length, 0), x=length, y=length+2)
            input_op = tf.nn.embedding_lookup(self._create_embedding_matrix(), input_op)
            print("input_op:{}".format(input_op))
            print("length_op:{}".format(length))
            output = self._rnn(input_op, length)
            output = tf.reshape(output, tuple(input_shape[:-1]) + (tf_util.get_shape(output)[-1], ))
            return output
        return embedding_layer


class BiRNNCharacterEmbedding(AbstractRNNCharacterEmbedding):
    def _rnn(self, input_op, input_length):
        cell_fw = self._rnn_cell()
        cell_bw = self._rnn_cell()
        initial_state_fw = cell_fw.zero_state(tf_util.get_shape(input_op)[0], dtype=tf.float32)
        initial_state_bw = cell_bw.zero_state(tf_util.get_shape(input_op)[0], dtype=tf.float32)
        _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_op,
                                                    sequence_length=input_length,
                                                    initial_state_fw=initial_state_fw,
                                                    initial_state_bw=initial_state_bw,
                                                    dtype=tf.float32, swap_memory=True)
        return tf.concat(states, axis=1)


    @abc.abstractmethod
    def _rnn_cell(self) -> tf.nn.rnn_cell.BasicRNNCell:
        pass


    def __init__(self, token_set: set, n_gram, embedding_shape=300):
        super().__init__(token_set, n_gram, embedding_shape)


class BiGRUCharacterEmbedding(BiRNNCharacterEmbedding):
    def _rnn_cell(self) -> tf.nn.rnn_cell.BasicRNNCell:
        return tf.nn.rnn_cell.GRUCell(self.embedding_shape)

    def __init__(self, token_set: set, n_gram, embedding_shape=300):
        super().__init__(token_set, n_gram, embedding_shape)


def load_character_vocabulary(embedding_type, n_gram, embedding_shape, token_list) -> CharacterEmbedding:
    embedding_dict = {"bigru": BiGRUCharacterEmbedding}
    token_set = set(more_itertools.collapse(token_list))
    return embedding_dict[embedding_type](token_set, n_gram, embedding_shape)

