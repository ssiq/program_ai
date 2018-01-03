import tensorflow as tf
import numpy as np


class WordMap(object):
    def __init__(self):
        self._end_label = "<END>"
        self._start_label = "<START>"
        self._identifier_label = "<IDENTIFIER>"
        self._placeholder_label = '<PLACEHOLDER>'

    @property
    def end_label(self):
        return self._end_label

    @property
    def start_label(self):
        return self._start_label

    @property
    def identifier_label(self):
        return self._identifier_label

    @property
    def placeholder_label(self):
        return self._placeholder_label


class KeyWordMap(WordMap):
    def __init__(self):
        from code_data.constants import pre_defined_cpp_token
        super().__init__()
        self._keyword = pre_defined_cpp_token | {self._start_label, self._end_label, self._identifier_label, self._placeholder_label}
        self._keyword = sorted(self._keyword)
        self._id_to_keyword = dict(list(enumerate(self._keyword)))
        self._keyword_to_id = {value:key for key, value in self._id_to_keyword.items()}

    def __len__(self):
        return len(self._keyword)

    def __getitem__(self, item):
        if item in self._keyword_to_id:
            return self._keyword_to_id[item]
        else:
            return self._keyword_to_id[self.identifier_label]


class Vocabulary(object):
    def __init__(self, word_id_map: WordMap, embedding_size):
        self.word_id_map = word_id_map
        self._embedding_matrix = np.random.randn(len(word_id_map), embedding_size)

    def word_to_id(self, word):
        return self.word_id_map[word]

    def id_to_word(self, id):
        if id in self.word_id_map._id_to_keyword:
            return self.word_id_map._id_to_keyword[id]
        else:
            return None

    @property
    def start_label(self):
        return self.word_id_map.start_label

    @property
    def end_label(self):
        return self.word_id_map.end_label

    @property
    def identifier_label(self):
        return self.word_id_map.identifier_label

    @property
    def placeholder_label(self):
        return self.word_id_map.placeholder_label

    @property
    def embedding_matrix(self):
        return self._embedding_matrix

    def create_embedding_layer(self):
        with tf.variable_scope("word_embedding_op"):
            _tf_embedding = tf.Variable(name="embedding", initial_value=self._embedding_matrix,
                                        dtype=tf.float32, trainable=True)
        def embedding_layer(input_op):
            """
            :param input_op: a tensorflow tensor with shape [batch, max_length] and type tf.int32
            :return: a looked tensor with shape [batch, max_length, embedding_size]
            """
            output = tf.nn.embedding_lookup(_tf_embedding, input_op)
            print("word embedding:{}".format(output))
            print("word_embedding_matrix:{}".format(_tf_embedding))
            return output

        return embedding_layer

    def parse_text(self, texts):
        """
        :param texts: a list of list of token
        :return:
        """
        max_text = max(map(lambda x:len(x), texts))
        texts = [[self.word_to_id(token) for token in text] for text in texts]
        texts = [text+[0]*(max_text-len(text)) for text in texts]
        return texts

    def parse_text_without_pad(self, texts, position_label=False):
        """
        :param texts: a list of string to pad
        :param position_label: a bool parameter indicating whether add START and END label
        :return: the padded texts
        """
        if position_label:
            texts = map(lambda x: [self.word_id_map.start_label] + x + [self.word_id_map.end_label] , texts)
        # texts = [[self.word_to_id(token) for token in text] for text in texts]
        new_string_list = []
        for l in texts:
            token_list = []
            try:
                for token in l:
                    token_list.append(self.word_to_id(token))
            except Exception as e:
                token_list = None
            new_string_list.append(token_list)
        return new_string_list


def load_vocabulary(word_vector_name, embedding_size) -> Vocabulary:
    namd_embedding_dict = {"keyword": KeyWordMap()}
    return Vocabulary(namd_embedding_dict[word_vector_name], embedding_size)
