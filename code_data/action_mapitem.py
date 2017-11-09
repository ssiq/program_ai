import json

class MAPITEM:

    def __init__(self, act_type=-1, from_char='', to_char='', err_pos=-1, ac_pos=-1):
        self.act_type = act_type
        self.from_char = from_char
        self.to_char = to_char
        self.ac_pos = ac_pos
        self.err_pos = err_pos

    def to_json_str(self):
        item = {}
        item['act_type'] = self.act_type
        item['from_char'] = self.from_char
        item['to_char'] = self.to_char
        item['ac_pos'] = self.ac_pos
        item['err_pos'] = self.err_pos
        res = json.dumps(item)
        return res

    def __dict__(self):
        di = {}
        di['act_type'] = self.act_type
        di['from_char'] = self.from_char
        di['to_char'] = self.to_char
        di['ac_pos'] = self.ac_pos
        di['err_pos'] = self.err_pos
        return di

    def __repr__(self):
        return str(self.__dict__())

    def get_ac_pos(self):
        return self.ac_pos

    def get_err_pos(self):
        return self.err_pos


class ACTION_MAPITEM(MAPITEM):

    def __init__(self, act_type=-1, from_char='', to_char='', ac_pos=-1):
        super().__init__(act_type=act_type, from_char=from_char, to_char=to_char, err_pos=-1, ac_pos=ac_pos)


class ERROR_CHARACTER_MAPITEM(MAPITEM):

    def __init__(self, act_type=-1, from_char='', to_char='', err_pos=-1, ac_pos=-1):
        super().__init__(act_type=act_type, from_char=from_char, to_char=to_char, err_pos=err_pos, ac_pos=ac_pos)


if __name__ == '__main__':
    err = ERROR_CHARACTER_MAPITEM()

