import ply.lex as lex

# Token list
tokens = (
    'LET', 'DEF', 'IDENTIFIER', 'NUMBER', 
    'PLUS', 'EQUALS'
)

# Simple tokens
t_PLUS = r'\+'
t_EQUALS = r'='
t_ignore = ' \t'

def t_LET(t):
    r'let'
    print(f"KEYWORD 'let' at line {t.lineno}")
    return t

def t_DEF(t):
    r'def'
    print(f"KEYWORD 'def' at line {t.lineno}")
    return t

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    print(f"IDENTIFIER '{t.value}' at line {t.lineno}")
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    print(f"NUMBER {t.value} at line {t.lineno}")
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

# Test it
data = '''
let x = 5
def add a b = a + b
'''

print("=== PYTHON LEXER DEMO ===")
lexer.input(data)
for token in lexer:
    print(f"Token: {token.type}, Value: {token.value}")