from code_data.code_preprocess import create_error_code


if __name__ == '__main__':
    code = r'''
    #include <iostream>
    int main() {
	double x, y;
	std::cin >> x >> y;
	std::cout << std::pow(x, y);
    }
    '''

    # for i in range(1000):
    #     print(i)
    #     code, error_code, action_maplist, error_character_maplist, error_count = create_error_code(code, error_count_range=(1, 1))
    code, error_code, action_maplist, error_character_maplist, error_count = create_error_code(code, error_count_range=(1, 5))
    print(code)
    print(error_code)
    for act in action_maplist:
        print(act)