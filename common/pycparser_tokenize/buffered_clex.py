from common.pycparser_tokenize.pycparser.pycparser.c_lexer import CLexer
from common.pycparser_tokenize.pycparser.pycparser.ply.lex import TOKEN
from common.util import maintain_function_co_firstlineno


class BufferedCLex(CLexer):
    def __init__(self, error_func, on_lbrace_func, on_rbrace_func, type_lookup_func, add_history_fn=lambda: None,
                 is_in_system_header=lambda x: False):
        super().__init__(error_func, on_lbrace_func, on_rbrace_func, type_lookup_func)
        self._tokens_buffer = []
        self._tokens_index = 0
        self._add_history_fn = add_history_fn
        self._is_in_system_header = is_in_system_header

    @property
    def add_history_fn(self):
        return self._add_history_fn

    @add_history_fn.setter
    def add_history_fn(self, _add_history_fn):
        self._add_history_fn = _add_history_fn

    @property
    def is_in_system_header(self):
        return self._is_in_system_header

    @is_in_system_header.setter
    def is_in_system_header(self, _is_in_system_header):
        self._is_in_system_header = _is_in_system_header

    def token(self):
        if self._tokens_index < len(self._tokens_buffer):
            self.last_token = self._tokens_buffer[self._tokens_index][0]
            self.filename = self._tokens_buffer[self._tokens_index][1]
            self._tokens_index += 1
        else:
            self.last_token = None

        if self.last_token is not None:
            if not self.is_in_system_header(self.last_token.lineno):
                self.add_history_fn()

        if self.last_token is not None:
            if self.last_token.type == 'LBRACE':
                self.on_lbrace_func()
            elif self.last_token.type == 'RBRACE':
                self.on_rbrace_func()
            elif self.last_token.type == 'ID' and self.type_lookup_func(self.last_token.value):
                self.last_token.type = 'TYPEID'

        return self.last_token

    def _all_tokens(self):
        def token():
            while True:
                tok = self.lexer.token()
                if not tok:
                    break
                else:
                    yield (tok, self.filename)
        return list(token())

    def input(self, text):
        super().input(text)
        self._tokens_buffer = self._all_tokens()
        # print(self._tokens_buffer)
        self._tokens_index = 0
        self.filename = ''

    @TOKEN(r'\}')
    @maintain_function_co_firstlineno(CLexer.t_LBRACE)
    def t_RBRACE(self, t):
        return t

    @TOKEN(r'\{')
    @maintain_function_co_firstlineno(CLexer.t_RBRACE)
    def t_LBRACE(self, t):
        return t

    @TOKEN(CLexer.identifier)
    @maintain_function_co_firstlineno(CLexer.t_ID)
    def t_ID(self, t):
        t.type = self.keyword_map.get(t.value, "ID")
        return t
