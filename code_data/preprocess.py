import itertools
import functools

_BEGIN_LABEL = '<BEGIN>'
_END_LABEL = '<END>'


class CharacterSet(object):
    def __init__(self, text_list):
        """
        :param text_list: a string list
        """
        character_set = functools.reduce(lambda a, b: set(a) | set(b), text_list, {})
        character_set.add(_BEGIN_LABEL)
        character_set.add(_END_LABEL)
        self.character_to_id_dict = dict((b, a) for a, b in enumerate(character_set, start=1))
        self.id_to_character_dict = dict(enumerate(character_set, start=1))
        self._character_set = character_set

    def parse_text(self, text_list):
        """
        :param text_list: a str list
        :return: a persed list of id's list
        """
        def parse(text):
            text = list(text)
            res = [self.character_to_id(_BEGIN_LABEL)]
            for i in map(lambda x: self.character_to_id(x), text):
                res.append(i)
            res.append(self.character_to_id(_END_LABEL))
            return res

        return list(map(lambda x: parse(x), text_list))

    def align_texts_with_same_length(self, ids):
        """
        :param ids: a list of id's list
        :return:
        """
        max_length = len(functools.reduce(lambda x,y: x if len(x)> len(y) else y, ids, []))
        return list(map(lambda x: x + [0]*(max_length - len(x)), ids))


    def character_to_id(self, c):
        return self.character_to_id_dict[c]

    def id_to_character(self, i):
        return self.id_to_character_dict[i]

    def character_set_size(self):
        return len(self.character_to_id_dict)

    @property
    def character_set(self):
        return self._character_set


if __name__ == "__main__":
    strings = ['ababvd', 'abccrf', 'aaaaaahhhhh']
    cs = CharacterSet(strings)
    print(cs.character_set)
    print(cs.parse_text(strings))
    print(cs.align_texts_with_same_length(cs.parse_text(strings)))




