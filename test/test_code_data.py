from code_data.error_action_reducer import *
import unittest
from common.new_tokenizer import tokenize


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._code = r'''
    #include <iostream>
    int main() {
	double x, y;
	std::cin >> x >> y;
	std::cout << std::pow(x, y);
    }
    '''
        cls._tokens = tokenize(cls._code)

    def test_identifier_position_random(self):
        _, pos = identifier_position_random(self._tokens)
        not_in_keyword = self._tokens[pos].value not in pre_defined_cpp_token
        self.assertTrue(not_in_keyword)

