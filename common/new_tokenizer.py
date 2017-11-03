import ply.lex as lex
from ply.lex import TOKEN
import sys
import re
from common import util

keywords = util.reverse_dict({
    "TOK_ASM": "asm",
    "TOK_AUTO": "auto",
    "TOK_BREAK": "break",
    "TOK_BOOL": "bool",
    "TOK_CASE": "case",
    "TOK_CATCH": "catch",
    "TOK_CDECL": "cdecl",
    "TOK_CHAR": "char",
    "TOK_CLASS": "class",
    "TOK_CONST": "const",
    "TOK_CONST_CAST": "const_cast",
    "TOK_CONTINUE": "continue",
    "TOK_DEFAULT": "default",
    "TOK_DELETE": "delete",
    "TOK_DO": "do",
    "TOK_DOUBLE": "double",
    "TOK_DYNAMIC_CAST": "dynamic_cast",
    "TOK_ELSE": "else",
    "TOK_ENUM": "enum",
    "TOK_EXPLICIT": "explicit",
    "TOK_EXPORT": "export",
    "TOK_EXTERN": "extern",
    "TOK_FALSE": "false",
    "TOK_FLOAT": "float",
    "TOK_FOR": "for",
    "TOK_FRIEND": "friend",
    "TOK_GOTO": "goto",
    "TOK_IF": "if",
    "TOK_INLINE": "inline",
    "TOK_INT": "int",
    "TOK_LONG": "long",
    "TOK_MUTABLE": "mutable",
    "TOK_NAMESPACE": "namespace",
    "TOK_NEW": "new",
    "TOK_OPERATOR": "operator",
    "TOK_PASCAL": "pascal",
    "TOK_PRIVATE": "private",
    "TOK_PROTECTED": "protected",
    "TOK_PUBLIC": "public",
    "TOK_REGISTER": "register",
    "TOK_REINTERPRET_CAST": "reinterpret_cast",
    "TOK_RETURN": "return",
    "TOK_SHORT": "short",
    "TOK_SIGNED": "signed",
    "TOK_SIZEOF": "sizeof",
    "TOK_STATIC": "static",
    "TOK_STATIC_CAST": "static_cast",
    "TOK_STRUCT": "struct",
    "TOK_SWITCH": "switch",
    "TOK_TEMPLATE": "template",
    "TOK_THIS": "this",
    "TOK_THROW": "throw",
    "TOK_TRUE": "true",
    "TOK_TRY": "try",
    "TOK_TYPEDEF": "typedef",
    "TOK_TYPEID": "typeid",
    "TOK_TYPENAME": "typename",
    "TOK_UNION": "union",
    "TOK_UNSIGNED": "unsigned",
    "TOK_USING": "using",
    "TOK_VIRTUAL": "virtual",
    "TOK_VOID": "void",
    "TOK_VOLATILE": "volatile",
    "TOK_WCHAR_T": "wchar_t",
    "TOK_WHILE": "while",
})


ANY = r'.'
NL = r"\n"
NOTNL = r"[^\n]"
BACKSL= r"\\"
ALNUM = r"[A-Za-z_0-9]"
DIGIT = r"[0-9]"
HEXDIGIT = r"[0-9A-Fa-f]"
DIGITS = r"{}+".format(DIGIT)
HEXDIGITS = r"{}+".format(HEXDIGIT)
SIGN = r"\+|-"
QUOTE = r"\""
LETTER = r"[A-Za-z_]"
STRCHAR = r"[^\"\n\\]"
BACKSL = r"\\"
ESCAPE = r"({}{})".format(BACKSL, ANY)
TICK = r"\'"
CCCHAR = r"[^\'\n\\]"

ELL_SUFFIX = r"[lL]([lL]?)"
INT_SUFFIX = r"([uU]("+ELL_SUFFIX+")?|("+ELL_SUFFIX+")[uU]?)"
FLOAT_SUFFIX = r"[flFL]"
INT = r"([1-9][0-9]*{}?)|([0][0-7]*{}?)|([0][xX][0-9A-Fa-f]+{}?)".format(INT_SUFFIX, INT_SUFFIX, INT_SUFFIX)
FLOAT = r"{}\.{}?([eE]{}?{})?{}?".format(DIGITS, DIGITS, SIGN, DIGITS, FLOAT_SUFFIX) + r"|" +\
        r"{}\.?([eE]{}?{})?{}?".format(DIGITS, SIGN, DIGITS, FLOAT_SUFFIX) + r"|" + \
        r"\.{}([eE]{}?{})?{}?".format(DIGITS, SIGN, DIGITS, FLOAT_SUFFIX)
STRING = r"L?{}({}|{})*{}".format(QUOTE, STRCHAR, ESCAPE, QUOTE)
CHARACTER = r"L?{}({}|{})*{}".format(TICK, CCCHAR, ESCAPE, TICK)
WHITESPACE = r"[\t\n\f\v\r ]+"
LINE_COMMENT = r"//{}*".format(NOTNL)
MULTILINE_COMMENT = r"/\*([^\*]|\**[^/])*\*+/"


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = keywords.get(t.value,'ID')    # Check for reserved words
    return t


operators = {
    "TOK_LPAREN": r"(",
    "TOK_RPAREN": r")",
    "TOK_LBRACKET": r"[",
    "TOK_RBRACKET": r"]",
    "TOK_ARROW": r"->",
    "TOK_COLONCOLON": r"::",
    "TOK_DOT": r".",
    "TOK_BANG": r"!",
    "TOK_TILDE": r"~",
    "TOK_PLUS": r"+",
    "TOK_MINUS": r"-",
    "TOK_PLUSPLUS": r"++",
    "TOK_MINUSMINUS": r"--",
    "TOK_AND": r"&",
    "TOK_STAR": r"*",
    "TOK_DOTSTAR": r".*",
    "TOK_ARROWSTAR": r"->*",
    "TOK_SLASH": r"/",
    "TOK_PERCENT": r"%",
    "TOK_LEFTSHIFT": r"<<",
    "TOK_RIGHTSHIFT": r">>",
    "TOK_LESSTHAN": r"<",
    "TOK_LESSEQ": r"<=",
    "TOK_GREATERTHAN": r">",
    "TOK_GREATEREQ": r">=",
    "TOK_EQUALEQUAL": r"==",
    "TOK_NOTEQUAL": r"!=",
    "TOK_XOR": r"^",
    "TOK_OR": r"|",
    "TOK_ANDAND": r"&&",
    "TOK_OROR": r"||",
    "TOK_QUESTION": r"?",
    "TOK_COLON": r":",
    "TOK_EQUAL": r"=",
    "TOK_STAREQUAL": r"*=",
    "TOK_SLASHEQUAL": r"/=",
    "TOK_PERCENTEQUAL": r"%=",
    "TOK_PLUSEQUAL": r"+=",
    "TOK_MINUSEQUAL": r"-=",
    "TOK_ANDEQUAL": r"&=",
    "TOK_XOREQUAL": r"^=",
    "TOK_OREQUAL": r"|=",
    "TOK_LEFTSHIFTEQUAL": r"<<=",
    "TOK_RIGHTSHIFTEQUAL": r">>=",
    "TOK_COMMA": r",",
    "TOK_ELLIPSIS": r"...",
    "TOK_SEMICOLON": r";",
    "TOK_LBRACE": r"{",
    "TOK_RBRACE": r"}",
}

for k, v in operators.items():
    setattr(sys.modules[__name__], 't_{}'.format(k), re.escape(v))

identifier = {
    "TOK_FLOAT_LITERAL": FLOAT,
    "TOK_INT_LITERAL": INT,
    "TOK_STRING_LITERAL": STRING,
    "TOK_CHARACTER_LITERAL": CHARACTER,
}

for k, v in identifier.items():
    setattr(sys.modules[__name__], 't_{}'.format(k), v)

@TOKEN(LINE_COMMENT)
def t_LINE_COMMENT(t):
    pass

@TOKEN(MULTILINE_COMMENT)
def t_MULTILINE_COMMENT(t):
    pass

@TOKEN(WHITESPACE)
def t_WHITESPACE(t):
    pass

@TOKEN(r"\#include\<?({}+)\>?".format(NOTNL))
def t_INCLUDE(t):
    value = ['#include', ]
    if t.value[8] == '<':
        value += '<'
        begin = 9
    else:
        begin = 8

    if t.value[-1] == '>':
        value += [t.value[begin:-1]]
        value += '>'
    else:
        value += [t.value[begin:]]

    t.value = value

    return t

def t_error(t):
    return t

tokens = ["LINE_COMMENT", "MULTILINE_COMMENT", "WHITESPACE", "INCLUDE", "ID"] + list(keywords.values()) + list(operators.keys()) + list(
    identifier.keys())

lexer = lex.lex(debug=True)

def tokenize(s):
    lexer.input(s)
    def token():
        while True:
            tok = lexer.token()
            if not tok:
                break
            else:
                yield tok
    return list(token())

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        source = f.read()
    for t in tokenize(source):
        print(t)