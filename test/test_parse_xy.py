from experiment.token_level_multirnn_on_fake_cpp_data import parse_xy, sample
from code_data.token_level_fake_code import GetTokens

if __name__ == '__main__':
    train, test, vaild = sample()
    from code_data.constants import char_sign_dict
    from embedding.wordembedding import load_vocabulary
    from embedding.character_embedding import load_character_vocabulary

    key_val = load_vocabulary('keyword', embedding_size=200)
    char_voc = load_character_vocabulary('bigru', n_gram=1, embedding_shape=100, token_list=char_sign_dict.keys())
    parse_xy_param = [key_val, char_voc]
    res = parse_xy(train, *parse_xy_param)

    token_id_list, token_id_length_list, character_id_list, character_id_length_list, is_continue_list, position_list, is_copy_list, keyword_list, copy_list = res

    record_id = 122943
    print(token_id_list.index)

    for record_id in token_id_list.index:

        i = 0
        token_ids = token_id_list.loc[record_id]
        token_id_lengths = token_id_length_list.loc[record_id]
        is_continues = is_continue_list.loc[record_id]
        positions = position_list.loc[record_id]
        is_copys = is_copy_list.loc[record_id]
        keywords = keyword_list.loc[record_id]
        copys = copy_list.loc[record_id]

        final_token_id = token_ids[0]

        while True:

            is_continue = is_continues[i]
            position = positions[i]
            is_copy = is_copys[i]
            keyword_id = keywords[i]
            copy_id = copys[i]

            if position % 2 == 0:
                if is_copy == 0:
                    final_token_id.insert(int(position/2), keyword_id)
                elif is_copy == 1:
                    tmp_token_id = final_token_id[copy_id]
                    final_token_id.insert(int(position/2), tmp_token_id)
            elif position % 2 == 1:

                if is_copy == 0 and keyword_id == key_val.word_to_id(key_val.placeholder_label):
                    final_token_id.pop(int(position/2))
                elif is_copy == 0:
                    final_token_id[int(position/2)] = keyword_id
                elif is_copy == 1:
                    tmp_token_id = final_token_id[copy_id]
                    final_token_id[int(position/2)] = tmp_token_id

            if is_continues[i] == 0:
                break

            i += 1

        token_true_list = [key_val.word_to_id(o.name) for o in GetTokens(train.loc[record_id, 'ac_code'])]

        # print(token_true_list)
        token_name = []
        for token_id in final_token_id:
            token_name.append(token_id)
        # print(token_name)

        assert len(token_name) == len(token_true_list)
        for i in range(len(token_name)):
            assert token_name[i] == token_true_list[i]
