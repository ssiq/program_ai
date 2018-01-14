import unittest
from experiment.experiment_util import sample, create_embedding, parse_xy_with_identifier_mask, \
    create_token_identify_mask, get_token_list, create_error_list, create_identifier_mask, parse_xy_token_level, sample_on_random_token_code_records, create_name_list_by_LexToken
from common.beam_search_util import flat_list, find_copy_input_position
from common.new_tokenizer import tokenize
import numpy as np
import random


class TestParseXY(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.keyword_voc, self.char_voc = create_embedding()
        # self.train, self.test, self.vaild = sample()


    def test_parse_xy_with_identifier_mask(self):
        parse_xy_param = [self.keyword_voc, self.char_voc, 5, 1]
        res = parse_xy_with_identifier_mask(self.train, 'train', *parse_xy_param)

        iden_id = self.keyword_voc.word_to_id(self.keyword_voc.identifier_label)

        identifier_masks = res[4]
        token_ids = res[0]
        for ind in identifier_masks.index:
            token_id_list = flat_list(token_ids.loc[ind])
            iden_mask_list = flat_list(identifier_masks.loc[ind])
            for token_id, iden_mask in zip(token_id_list, iden_mask_list):
                if token_id != iden_id:
                    self.assertEqual(np.sum(iden_mask), 0)
                else:
                    self.assertGreater(np.sum(iden_mask), 0)

    def test_parse_xy_token_level(self):
        train, test, vaild = sample_on_random_token_code_records()
        # train = train.iloc[0:1]
        parse_xy_param = [self.keyword_voc, self.char_voc, 5, 1]
        res = parse_xy_token_level(train, 'train', *parse_xy_param)
        # print(train['ac_code'].iloc[0])
        # print(train['code'].iloc[0])
        # print(train['action_character_list'].iloc[0])
        # for r in res:
        #     print(r)

        token_id_list, token_len_list, char_id_list, char_len_list, iden_list, is_continue_list, position_list, is_copy_list, keyword_list, copy_id_list = res
        res_len = len(token_id_list.index)

        for k in range(100):
            l = random.randint(0, res_len-1)
            cur_len = len(token_id_list.iloc[l])
            if cur_len == 1:
                continue
            i = random.randint(0, cur_len-2)
            reduce_token_id = self.do_one_recover_iterator(token_id_list.iloc[l][i], token_len_list.iloc[l][i], char_id_list.iloc[l][i],
                                         char_len_list.iloc[l][i], iden_list.iloc[l][i], is_continue_list.iloc[l][i],
                                         position_list.iloc[l][i], is_copy_list.iloc[l][i], keyword_list.iloc[l][i], copy_id_list.iloc[l][i])
            self.assertEqual(reduce_token_id, token_id_list.iloc[l][i+1])

        for k in range(100):
            l = random.randint(0, res_len - 1)
            cur_index = token_id_list.index[l]
            ac_code = train['ac_code'].loc[cur_index]
            ac_objs = tokenize(ac_code)
            name_list = create_name_list_by_LexToken(ac_objs)
            id_list = self.keyword_voc.parse_text_without_pad([name_list], False)[0]
            pre_token_id = self.recovery_token_id(token_id_list.iloc[l], token_len_list.iloc[l], char_id_list.iloc[l],
                                         char_len_list.iloc[l], iden_list.iloc[l], is_continue_list.iloc[l],
                                         position_list.iloc[l], is_copy_list.iloc[l], keyword_list.iloc[l], copy_id_list.iloc[l])

            self.assertEqual(pre_token_id, id_list)

        iden_id = self.keyword_voc.word_to_id(self.keyword_voc.identifier_label)

        identifier_masks = res[4]
        token_ids = res[0]
        for ind in identifier_masks.index:
            token_id_list = flat_list(token_ids.loc[ind])
            iden_mask_list = flat_list(identifier_masks.loc[ind])
            for token_id, iden_mask in zip(token_id_list, iden_mask_list):
                if token_id != iden_id:
                    self.assertEqual(np.sum(iden_mask), 0)
                else:
                    self.assertGreater(np.sum(iden_mask), 0)




    def test_create_identifier_mask(self):
        keyword_set = {'3', '5', '1', '2', '4'}
        tokens = ['1', '6', '3', '8', '7', '7', '9', '4', '2', '5', '0', '6']
        masked = [[0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0]]
        mask_dict = {'0': 1, '6': 2, '7': 3, '8': 4, '9': 5}
        res, res_dict = create_identifier_mask(tokens, keyword_set)
        self.assertEqual(res, masked)
        self.assertEqual(res_dict, mask_dict)

    def test_create_token_identify_mask(self):
        keyword_set = {'3', '5', '1', '2', '4'}
        token_name_list = [['1', '6', '3', '8', '7', '7', '9', '4', '2', '5', '0', '6'],
                           ['3', '7', '1', '1', '0', '9']]
        masked = [
            [[0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0]],
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1]]
        ]
        mask_dict = [{'0': 1, '6': 2, '7': 3, '8': 4, '9': 5},
                     {'0': 1, '6': 2, '7': 3, '8': 4, '9': 5}]
        one = {'token_name_list': token_name_list}
        one = create_token_identify_mask(one, pre_defined_token_set=keyword_set)
        self.assertEqual(one['token_identify_mask'], masked)
        self.assertEqual(one['copy_name_list'], mask_dict)

        token_name_list = [['1', '6', '3', '8', '7', '7', '9', '4', '2', '5', '0', '6'],
                           ['3', '7', '1', '1', '0', '12']]
        one = {'token_name_list': token_name_list}
        one = create_token_identify_mask(one, pre_defined_token_set=keyword_set)
        self.assertIsNone(one['res'])

    def recovery_token_id(self, *args):
        # token_id, token_len, char_id, char_len, iden, is_continue, position, is_copy, keyword, copy_id = args
        first_token_id = args[0][0]
        for token_id, token_len, char_id, char_len, iden, is_continue, position, is_copy, keyword_id, copy_id in zip(*args):
            first_token_id = self.do_one_recover_iterator(first_token_id, token_len, char_id, char_len, iden, is_continue, position, is_copy, keyword_id, copy_id)
        return first_token_id

    def do_one_recover_iterator(self, token_id, token_len, char_id, char_len, iden, is_continue, position, is_copy, keyword_id, copy_id):
        word_id = -1
        plh_id = self.keyword_voc.word_to_id(self.keyword_voc.placeholder_label)
        if is_copy:
            copy_pos = find_copy_input_position(iden, copy_id)
            word_id = token_id[copy_pos]
        else:
            word_id = keyword_id
        if position % 2 == 0:
            position = int(position / 2)
            token_id = token_id[0:position] + [word_id] + token_id[position:]
        elif word_id == plh_id:
            position = int(position / 2)
            token_id = token_id[0:position] + token_id[position + 1:]
        else:
            position = int(position / 2)
            token_id = token_id[0:position] + [word_id] + token_id[position + 1:]
        return token_id

