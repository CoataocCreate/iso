import asyncio
import ctypes
import hashlib
import math
import os
import random
import sys
import typing
from colorama import Fore, Style, init; init(autoreset=True)
import re
import base64
import aiofiles
from ctypes import c_int, windll, c_char_p
from pathlib import Path
import numpy as np
from ursina import *
sys.set_int_max_str_digits(100000000)

# -------------------------------
#            C DLLS
# -------------------------------

dll_path = Path(__file__).parent / "sysp.dll"
clib = windll.LoadLibrary(str(dll_path.resolve()))
clib.sysp.argtypes = [c_char_p, c_int]
clib.sysp.restype = None

# -------------------------------
#   Standard Library Definition
# -------------------------------

debug = False

# -------------------------------
#          Token Definition
# -------------------------------
class Token:
    """Represents a token produced by the lexer."""
    def __init__(self, type_: str, value: typing.Any = None, pos=None, sstate=None) -> None:
        self.type = type_
        self.value = value
        self.pos = pos
        self.sstate = sstate

    def __repr__(self) -> str:
        if self.value is not None:
            return f"Token({self.type}, {repr(self.value)})"
        return f"Token({self.type})"

# -------------------------------
#             Lexer
# -------------------------------
prefixers = 'ruls^p8db-z/'

def rp(tar, x, y, sel=None):
    if sel is not None:
        sel = int(sel)
        # Check if the index is valid
        if 0 <= sel < len(tar):
            # Replace at the specific position and combine the result
            tar = tar[:sel] + tar[sel:].replace(x, y, 1)
        return tar
    else:
        return tar.replace(x, y)
    
def each(op, operand, lst):
    if op == "+":
        return [x + operand for x in lst]
    elif op == "-":
        return [x - operand for x in lst]
    elif op == "*":
        return [x * operand for x in lst]
    elif op == "/":
        return [x / operand for x in lst] if operand != 0 else "Error: Division by zero"
    elif op == "//":
        return [x // operand for x in lst] if operand != 0 else "Error: Division by zero"
    elif op == "%":
        return [x % operand for x in lst]
    elif op == "**":
        return [x ** operand for x in lst]
    else:
        return "Error: Unsupported operator"
    
def flat(lst):
    """Flattens a nested list."""
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flat(item))
        else:
            flattened.append(item)
    return flattened

def rdict(d, reverse_type="keys"):
    """Reverses the dictionary by either keys or values."""
    if reverse_type == "keys":
        # Reversing the keys (keeping values in place)
        return {k[::-1]: v for k, v in d.items()}
    elif reverse_type == "values":
        # Reversing the values (keeping keys in place)
        return {k: v[::-1] for k, v in d.items()}
    else:
        return "Error: Invalid reverse type. Use 'keys' or 'values'."


def DEL(x):
    del x

def compute(x):
    lexer = Lexer(x)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    tree = parser.parse()
    interpreter = Interpreter()
    result = None
    for stmt in tree:
        result = interpreter.visit(stmt)
    return result

def ifp(x, expr, v, t,e=None):
    condition = eval(f"{x} {expr.replace('T', '==')} {v.replace('$x', str(x))}")
    return compute(t.replace('$x', str(x))) if condition else compute(e.replace('$x', str(x)))

extended_transforms = {
    'ast': lambda x: f'{Parser(Lexer(x).tokenize()).parse()}',
    'lex': lambda x: f'{Lexer(x).tokenize()}',
    'r': lambda x: x[::-1],
    '^': lambda x: x[:1].upper() + x[1:],
    'l': lambda x: x.lower(),
    'u': lambda x: x.upper(),
    'len': lambda x: str(len(x)),
    's': lambda x: x.strip(),                                   # strip spaces
    '8': lambda x: base64.b64encode(x.encode()).decode(),
    'd': lambda x: ''.join(ch for ch in x if not ch.isdigit()),
    'b': lambda x: ' '.join(format(ord(c), 'b') for c in x),
    '-': lambda x: ''.join(dict.fromkeys(x)),
    'z': lambda x: ''.join(random.sample(x, len(x))),
    'hex': lambda x: x.encode().hex(),
    'hash': lambda x: hashlib.sha256(x.encode()).hexdigest(),
    'eval': lambda x: str(eval(x)),
    'rp': lambda value, x, y, sel=None: rp(value, x, y, sel),
    'cp': lambda x: print(x),
    'int': lambda x: int(x),
    'type': lambda x: type(x),
    'str': lambda x: ''.join([str(c) for c in x]) if isinstance(x, list) else str(x),
    'bool': lambda x: bool(x),
    'fb': lambda x: ''.join(chr(int(b, 2)) for b in x.split()),
    'bin~dec': lambda x: str(int(x.replace(' ', ''), 2)),
    'bin~hex': lambda x: hex(int(x.replace(' ', ''), 2))[2:],
    'bin~oct': lambda x: oct(int(x.replace(' ', ''), 2))[2:],
    'bin~asc': lambda x: ''.join(chr(int(b, 2)) for b in x.split()),
    'unicode': lambda x: ' '.join(f'\\u{ord(c):04x}' for c in x),
    'con': lambda *args: ''.join(args),
    'utf8': lambda x: x.encode('utf-8'),
    'sha256': lambda x: hashlib.sha256(x.encode()).hexdigest(),
    'camel': lambda x: ''.join([word.capitalize() if i > 0 else word.lower() for i, word in enumerate(x.split())]),
    'dec~bin': lambda x: bin(int(x))[2:] if isinstance(x, int) or (x.isdigit() and int(x) >= 0) else 'Invalid Input',
    'hex~bin': lambda x: bin(int(x, 16))[2:] if all(c in '0123456789abcdefABCDEF' for c in x) else 'Invalid Hex',
    'asc~bin': lambda x: ' '.join(format(ord(c), '08b') for c in x),
    'each': lambda value, x, y: each(x, y, value),
    'list~f': lambda x: flat(x),
    'dict~r': lambda x, typ: rdict(x, typ), 
    'del': lambda x: DEL(x),
    'list': lambda x, y: [int(c) if y == 'int' else c for c in x],
    'math.prod': lambda x: np.prod(x, dtype=object),
    'math.sin': lambda x: math.sin(x),
    'math.cos': lambda x: math.cos(x),
    'recurb': lambda n: list(range(n, 0, -1)),
    'memory.addr': lambda obj: hex(id(obj)),
    'if': ifp
    }

tts = []
class Lexer:
    """Converts an input string into a stream of tokens."""
    def __init__(self, text: str, debug=None) -> None:
        self.text = text
        self.length = len(text)
        self.pos = 0
        self.current_char = self.text[self.pos] if self.pos < self.length else None
        self.debug = debug

    def advance(self) -> None:
        self.pos += 1
        pos = self.pos
        text = self.text  # Local variable for faster access
        length = self.length  # Local variable for faster access
        # Use local variables to avoid repeated attribute lookups
        if pos < length:
            self.current_char = text[pos]
        else:
            self.current_char = None

    def skip_whitespace(self) -> None:
        """Skip any whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self) -> None:
        """Skip characters until the end of the line."""
        while self.current_char is not None and self.current_char != '\n':
            self.advance()

    def number(self) -> Token:
        text = self.text
        pos = self.pos
        end = len(text)
        start_pos = pos

        num_chars = []
        has_dot = False  # Tracks if there's a decimal point in the number

        while pos < end:
            char = text[pos]
            if char.isdigit():
                num_chars.append(char)
            elif char == '.' and not has_dot:  # Allow only one dot
                num_chars.append(char)
                has_dot = True
            else:
                break
            pos += 1

        self.pos = pos
        self.current_char = text[pos] if pos < end else None

        num_str = ''.join(num_chars)
        if has_dot:  # If it contains a dot, treat it as a float
            return Token("FLOAT", float(num_str), start_pos)
        else:  # Otherwise, treat it as an integer
            return Token("NUMBER", int(num_str), start_pos)

    def string(self) -> Token:
        """Handle string literals without transformations."""
        start_pos = self.pos

        prefixers = []
        while self.current_char == '<':
            self.eat('<')
            prefix_content = []
            while self.current_char is not None and self.current_char != '>':
                prefix_content.append(self.current_char)
                self.advance()
            if self.current_char != '>':
                self.error("Unterminated <...> block")
            self.advance()  # Skip '>'
            prefixers.append(''.join(prefix_content).strip())

        if self.current_char not in ("'", '"', "`", "@"):
            self.error("Expected opening quote")
        opening_quote = self.current_char  # Save which quote was used
        self.advance()

        chars = []
        while self.current_char not in (None, opening_quote):
            chars.append(self.current_char)
            self.advance()
        if self.current_char != opening_quote:
            self.error("Unterminated string literal")
        self.advance()

        value = ''.join(chars)
        
        if prefixers:
         return Token("PREFIXER-STR", (':'.join(prefixers), value), start_pos)
        else:
            return Token("STRING", value, start_pos)

    def identifier(self) -> Token:
        """Handles identifiers, numbers, and keywords."""
        result = ""
        start_pos = self.pos
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char in "_$&|`?~;"):

            result += self.current_char
            self.advance()

        # Keywords
        keywords = {
            "print": "PRINT",
            "True": "BOOL",
            "False": "BOOL",
            "func": "FUNC",
            "return": "RETURN",
            "while": "WHILE",
            "if": "IF",
            "else": "ELSE",
            "elif": "ELIF",
            "import": "IMPORT",
            "None": "NONE",
            "class": "CLASS",
            "for": "FOR",
            "in": "IN",
            "local": "LOCAL",
            "const": "CONST",
            "end": "EOF"
        }

        if result in keywords:
            if result in ("True", "False"):
                return Token("BOOL", result == "True", start_pos)
            return Token(keywords[result], None, start_pos)

        # Entirely digits → NUMBER
        if result.isdigit():
            return Token("NUMBER", int(result), start_pos)

        # Otherwise → IDENTIFIER
        return Token("IDENTIFIER", result, start_pos)



    def error(self, message: str) -> None:
            """Raise a syntax error with context information."""
            line_number = self.text[:self.pos].count('\n') + 1
            line_start = self.text.rfind('\n', 0, self.pos) + 1
            line_end = self.text.find('\n', self.pos)
            if line_end == -1:
                line_end = len(self.text)
            line_content = self.text[line_start:line_end]
            col_number = self.pos - line_start + 1

            # Highlight the error visually
            pointer_line = " " * (col_number - 1) + "^"

            error_message = (
                f"\n{Fore.RED}error{Style.RESET_ALL}: {message}\n"
                f"  --> line {line_number}, column {col_number}\n"
                f"   |\n"
                f"{line_number:3}| {line_content}\n"
                f"   | {Fore.YELLOW}{pointer_line}{Style.RESET_ALL}\n"
            )
            raise SyntaxError(error_message)
        

    def get_next_token(self) -> Token:
        """Lex and return the next token in the input text."""

        strs = ("'", '"', '`', '@')

        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '#': 
                self.skip_comment()
                continue

            if self.current_char == '-':
                start_pos = self.pos
                self.advance()
                if self.current_char == '>':
                    self.advance()
                    return Token("ARROW", None, start_pos)
               
                if self.current_char == '*':
                        self.advance()
                        return Token("DARROW", None, start_pos)
                else:
                    tok = Token("MINUS", None, start_pos)
                    return tok

            if self.current_char in ['<', '|']:
                start_pos = self.pos
                self.advance()

                # Parse the content inside <...> or |...|
                content = []
                while self.current_char is not None and self.current_char not in ['>', '|']:
                    content.append(self.current_char)
                    self.advance()

                if self.current_char not in ['>', '|']:
                    self.error("Unterminated <...> or |...> block")

                self.advance()  # Skip the closing '>' or '|'

                # Check if this is a prefixer or a comparison operator
                if self.current_char.isalnum() or self.current_char in [' ', '\n']:
                    return Token("PREFIXER", ''.join(content).strip(), start_pos)
                else:
                    # Fallback to LT operator if not a prefixer
                    return Token("LT", None, start_pos)

            if self.current_char in strs:
             return self.string()
            
            if self.current_char in prefixers:
                peek_pos = self.pos + 1
                while peek_pos < self.length and self.text[peek_pos] in prefixers:
                    peek_pos += 1
                if peek_pos < self.length and self.text[peek_pos] in strs:
                    return self.string()
                else:
                    return self.identifier()
            
            if self.current_char.isalnum() or self.current_char in "_$&|`?~;":
               return self.identifier()

            if self.current_char == '+':
                tok = Token("PLUS", None, self.pos)
                self.advance()
                return tok

            if self.current_char == '*':
                tok = Token("MUL", None, self.pos)
                self.advance()
                return tok
            
            if self.current_char == '/':
                self.advance()
                return Token("DIV", None, self.pos)
            
            if self.current_char == '(':
                tok = Token("LPAREN", None, self.pos)
                self.advance()
                return tok
            if self.current_char == ')':
                tok = Token("RPAREN", None, self.pos)
                self.advance()
                return tok
            
            if self.current_char == '=':
                        self.advance()
                        if self.current_char == '=':  # Check for '=='
                            self.advance()
                            return Token("EQ", "==", self.pos)
                        return Token("ASSIGN", "=", self.pos)  # Single '=' for assignment
            
            if self.current_char == '%':
                tok = Token("MOD", None, self.pos)
                self.advance()
                return tok
            
            if self.current_char == '&':
                tok = Token("AND", None, self.pos)
                self.advance()
                return tok
            
            if self.current_char == '^':
                tok = Token("XOR", None, self.pos)
                self.advance()
                return tok
            
            if self.current_char == ',':
                tok = Token("COMMA", None, self.pos)
                self.advance()
                return tok
            
            if self.current_char == '{':
                tok = Token("LBRACE", None, self.pos)
                self.advance()
                return tok
            
            if self.current_char == '}':
                tok = Token("RBRACE", None, self.pos)
                self.advance()
                return tok

            if self.current_char == '[':
                tok = Token("LBRACKET", None, self.pos)
                self.advance()
                return tok
            
            if self.current_char == ']':
                tok = Token("RBRACKET", None, self.pos)
                self.advance()
                return tok
            
            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':  # Handle '>='
                    self.advance()
                    return Token("GE", None, self.pos)
                return Token("GT", None, self.pos)
            
            if self.current_char == '!':
                self.advance()
                if self.current_char == '=':  # Handle '!='
                    self.advance()
                    return Token("NE", None, self.pos)
                
            if self.current_char == '.':
                self.advance()
                return Token("DOT", None, self.pos)
            
            if self.current_char == ':':
                tok = Token("COLON", None, self.pos)
                self.advance()
                return tok
            
            self.error(f"Unexpected character '{self.current_char}'")
        
        return Token("EOF", None, self.pos)

    def tokenize(self) -> typing.List[Token]:
        """Tokenize the entire input text."""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == "EOF":
                break
        if self.debug:
         print(tokens)
        return tokens


# -------------------------------
#           AST Nodes
# -------------------------------
class AST:
    """Base class for all AST nodes."""
    pass

class PrintNode(AST):
    def __init__(self, exprs: typing.List[AST]) -> None:
        self.exprs = exprs  # List of expressions to print

    def __repr__(self) -> str:
        return f"PrintNode({self.exprs})"
    

class AssignNode(AST):
    """Represents variable assignment (e.g., x = 10)."""
    def __init__(self, var_name, expr):
        self.var_name = var_name
        self.expr = expr

    def __repr__(self):
        return f"AssignNode({self.var_name}, {self.expr})"


class Num(AST):
    def __init__(self, token: Token) -> None:
        self.token = token
        self.value = token.value

    def __repr__(self) -> str:
        return f"Num({self.value})"


class Str(AST):
    def __init__(self, token: Token) -> None:
        self.token = token
        self.value = token.value

    def __repr__(self) -> str:
        return f"Str({self.value})"
    
class ClassDef(AST):
    def __init__(self, class_name: str, body: typing.List[AST]) -> None:
        self.class_name = class_name
        self.body = body
    def __repr__(self) -> str:
        return f"ClassDef({self.class_name}, body={self.body})"
    
class ClassInstantiation(AST):
    def __init__(self, class_name: str, args: typing.List[AST]) -> None:
        self.class_name = class_name
        self.args = args
    def __repr__(self) -> str:
        return f"ClassInstantiation({self.class_name}, args={self.args})"
    
class List(AST):
    """Represents a list in the AST."""
    def __init__(self, elements: typing.List[AST]) -> None:
        self.elements = elements  # Store the list elements

    def __repr__(self) -> str:
        return f"List({self.elements})"
    
class ListIndexNode(AST):
    """Represents a list indexing operation (e.g., list[0])."""
    def __init__(self, list_name: str, index: AST) -> None:
        self.list_name = list_name
        self.index = index

    def __repr__(self) -> str:
        return f"ListIndexNode(list_name={self.list_name}, index={self.index})"
    
class Bool(AST):
    def __init__(self, token: Token) -> None:
        self.token = token
        self.value = token.value

    def __repr__(self) -> str:
        return f"Bool({self.value})"


class BinOp(AST):
    def __init__(self, left: AST, op: Token, right: AST) -> None:
        self.left = left
        self.op = op  # The operator token (PLUS, MINUS, etc.)
        self.right = right

    def __repr__(self) -> str:
        return f"BinOp({self.left}, {self.op.type}, {self.right})"


class VarNode(AST):
    """Represents a variable reference (e.g., using 'x' in an expression)."""
    def __init__(self, var_name):
        self.var_name = var_name

    def __repr__(self):
        return f"VarNode({self.var_name})"


class FunctionDef(AST):
    """Represents a function definition."""
    def __init__(self, func_name: str, parameters: typing.List[typing.Tuple[str, typing.Optional[AST]]], body: typing.List[AST]) -> None:
        self.func_name = func_name
        # Parameters now store tuples of (name, default_value). Default value is None if not provided.
        self.parameters = parameters
        self.body = body

    def __repr__(self) -> str:
        return f"FunctionDef({self.func_name}, params={self.parameters}, body={self.body})"

class FunctionCall(AST):
    """Represents a function call."""
    def __init__(self, func_name: str, args: typing.List[AST]) -> None:
        self.func_name = func_name
        self.args = args

    def __repr__(self) -> str:
        return f"FunctionCall({self.func_name}, args={self.args})"
    
class CompareNode(AST):
    """Represents a comparison operation (e.g., x < 5, a == b)."""
    def __init__(self, left: AST, op: Token, right: AST) -> None:
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self) -> str:
        return f"CompareNode({self.left}, {self.op.type}, {self.right})"
    
class WhileNode(AST):
    """Represents a while loop: while condition { body }"""
    def __init__(self, condition: AST, body: typing.List[AST]) -> None:
        self.condition = condition
        self.body = body

    def __repr__(self) -> str:
        return f"WhileNode(condition={self.condition}, body={self.body})"


class ReturnNode(AST):
    """Represents a return statement inside a function."""
    def __init__(self, expr: AST) -> None:
        self.expr = expr

    def __repr__(self) -> str:
        return f"ReturnNode({self.expr})"
    

class IfNode(AST):
    """Represents an if-elif-else statement."""
    def __init__(self, condition, true_branch, elif_branches=None, false_branch=None):
        self.condition = condition
        self.true_branch = true_branch
        self.elif_branches = elif_branches or []  # List of (condition, body) tuples
        self.false_branch = false_branch

    def __repr__(self):
        return (f"IfNode(condition={self.condition}, true_branch={self.true_branch}, "
                f"elif_branches={self.elif_branches}, false_branch={self.false_branch})")
    
class ImportNode(AST):
    """Represents an import statement."""
    def __init__(self, module_name):
        # this line must be here
        self.module_name = module_name

    def __repr__(self) -> str:
        return f"ImportNode(module_name={self.module_name})"

        
class NoneNode(AST):
    """Represents the None value."""
    def __init__(self):
        self.value = None

    def __repr__(self):
        return "NoneNode(None)"
    
class MethodCall(AST):
    """Represents a method call on any object (e.g., obj.method(arg1, arg2))."""
    def __init__(self, list_name: str, method_name: str, args: typing.List[AST]) -> None:
        self.list_name = list_name
        self.method_name = method_name
        self.args = args

    def __repr__(self) -> str:
        return f"MethodCall(object={self.list_name}, name={self.method_name}, args={self.args})"

class Dict(AST):
    """Represents a dictionary in the AST."""
    def __init__(self, elements: typing.List[typing.Tuple[AST, AST]]) -> None:
        self.elements = elements  # List of key-value pairs

    def __repr__(self) -> str:
        return f"Dict({self.elements})"
    
class ForNode(AST):
    """Represents a for loop: for variable in iterable { body }"""
    def __init__(self, variable: str, iterable: AST, body: typing.List[AST]) -> None:
        self.variable = variable
        self.iterable = iterable
        self.body = body

    def __repr__(self) -> str:
        return f"ForNode(variable={self.variable}, iterable={self.iterable}, body={self.body})"
    
class TransformNode(AST):
    """Represents a transformation applied by a prefixer."""
    def __init__(self, transform: str, value: AST):
        self.transform = transform  # The transformation(s) to apply (e.g., 'r', 'u', etc.)
        self.value = value          # The value to transform

    def __repr__(self):
        return f"TransformNode(transform={self.transform}, value={self.value})"
    
class ArrowNote(AST):
    """Represents a variable reference (e.g., using 'x' in an expression)."""
    def __init__(self, var_name, cvalue):
        self.var_name = var_name
        self.cvalue = cvalue

    def __repr__(self):
        return f"ArrowNode({self.var_name} -> {self.cvalue})"
    
class DArrowNote(AST):
    """Represents a variable reference (e.g., using 'x' in an expression)."""
    def __init__(self, var_name):
        self.var_name = var_name

    def __repr__(self):
        return f"DArrowNode({self.var_name})"
    
class AttributeAccess(AST):
    def __init__(self, owner: AST, name: str):
        self.owner = owner
        self.name  = name
    def __repr__(self):
        return f"{self.owner!r}.{self.name}"
    
class LocalAssignNode(AST):
    """Represents a local variable assignment within a function."""
    def __init__(self, var_name, expr):
        self.var_name = var_name
        self.expr = expr
    def __repr__(self):
        return f"LocalAssignNode({self.var_name}, {self.expr})"
    
class ConstAssignNode(AST):
    """Represents a constant assignment (e.g., const x = 10)"""
    def __init__(self, var_name, expr):
        self.var_name = var_name
        self.expr = expr
    def __repr__(self):
        return f"ConstAssignNode({self.var_name}, {self.expr})"

# -------------------------------
#             Parser
# -------------------------------
class Parser:
    """
    Implements a recursive descent parser.
    
    Extended grammar now includes:
    
        statement       -> PRINT expr
                           | IDENTIFIER ASSIGN expr
                           | func IDENTIFIER LPAREN [params] RPAREN LBRACE {statement} RBRACE
                           | RETURN expr
                           
        expr            -> term ((PLUS | MINUS) term)*
        term            -> factor ((MUL | DIV | MOD | AND | XOR) factor)*
        factor          -> NUMBER | STRING | BOOL | IDENTIFIER [function_call] | LPAREN expr RPAREN
    """
    def __init__(self, tokens: typing.List[Token], debug=None) -> None:
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.classes = {}
        self.debug = debug

    def error(self, message: str) -> None:
        token = self.current_token
        line_number = self.tokens[:self.pos].count(Token("NEWLINE")) + 1
        line_start = self.tokens[:self.pos].rfind(Token("NEWLINE")) + 1 if Token("NEWLINE") in self.tokens[:self.pos] else 0
        line_end = self.tokens[self.pos:].find(Token("NEWLINE")) + self.pos if Token("NEWLINE") in self.tokens[self.pos:] else len(self.tokens)
        line_content = ''.join(str(tok.value) for tok in self.tokens[line_start:line_end] if tok.value is not None)
        col_number = self.pos - line_start + 1

        # Highlight the error visually
        pointer_line = " " * (col_number - 1) + "^"

        error_message = (
            f"\n{Fore.RED}error{Style.RESET_ALL}: {message}\n"
            f"  --> line {line_number}, column {col_number}\n"
            f"   |\n"
            f"{line_number:3} | {line_content}\n"
            f"   | {Fore.YELLOW}{pointer_line}{Style.RESET_ALL}\n"
        )
        raise SyntaxError(error_message)

    def eat(self, token_type: str) -> None:
        current_token = self.current_token
        if current_token.type == token_type:
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
        else:
            self.error(f"Expected token {token_type}, got {current_token.type}")

    def import_statement(self):
        self.eat("IMPORT")
        module = self.current_token.value
        self.eat("IDENTIFIER")
        return ImportNode(module)
    
    def list_statement(self) -> List:
        """Parses a list statement: [expr, expr, ...]"""
        self.eat("LBRACKET")
        elements = []
        if self.current_token.type != "RBRACKET":
            elements.append(self.expr())
            while self.current_token.type == "COMMA":
                self.eat("COMMA")
                elements.append(self.expr())
        self.eat("RBRACKET")
        return List(elements)
    
    def dict_statement(self) -> Dict:
        """Parses a dictionary literal: {key: value, key: value, ...}"""
        self.eat("LBRACE")
        elements = []
        if self.current_token.type != "RBRACE":
            key = self.expr()
            self.eat("COLON")
            value = self.expr()
            elements.append((key, value))
            while self.current_token.type == "COMMA":
                self.eat("COMMA")
                key = self.expr()
                self.eat("COLON")
                value = self.expr()
                elements.append((key, value))
        self.eat("RBRACE")
        return Dict(elements)
    
    def arrow_statement(self, var_name: str) -> ArrowNote:
        self.eat("ARROW")       # eat '->'
        expr_node = self.expr()  # parse whatever comes next
        return ArrowNote(var_name, expr_node)
    
    def darrow_statment(self, var_name:str) -> DArrowNote:
        self.eat("DARROW")
        return DArrowNote(var_name)

    def apply_prefixers(self, value: str, prefixers: str) -> str:
        """Apply prefixer transformations to a string."""
        for prefix in prefixers.split(':'):
            if prefix in extended_transforms:
                value = extended_transforms[prefix](value)
        return value

    def factor(self) -> AST:
        """Parses numbers, strings, booleans, variables (or function calls), and expressions in parentheses."""
        token = self.current_token
        
        # Handle prefixers (<...>)
        prefixers = []
        while token.type == "PREFIXER":
            prefixers.append(token.value)
            self.eat("PREFIXER")
            token = self.current_token

        if token.type == "FLOAT":  # Add support for FLOAT tokens
            self.eat("FLOAT")
            return Num(token)

        if token.type == "NUMBER":
            self.eat("NUMBER")
            return Num(token)

        elif token.type == "STRING":
            self.eat("STRING")
            return Str(token)

        elif token.type == "BOOL":
            self.eat("BOOL")
            return Bool(token)
        
        elif token.type == "NONE":
            self.eat("NONE")
            return NoneNode()
    
        elif token.type == "IDENTIFIER":
            owner = VarNode(token.value)
            id = self.current_token.value
            self.eat("IDENTIFIER")

            if self.current_token.type == "ARROW":
               return self.arrow_statement(id)  # Call the new arrow_statement method
            
            elif self.current_token.type == "DARROW":
                return self.darrow_statment(id)
            
            elif self.current_token.type == "LPAREN":  # Function call
                self.eat("LPAREN")
                args = []
                if self.current_token.type != "RPAREN":
                    args.append(self.expr())
                    while self.current_token.type == "COMMA":
                        self.eat("COMMA")
                        args.append(self.expr())
                self.eat("RPAREN")
                if token.value in self.classes:
                    return ClassInstantiation(token.value, args)
                return FunctionCall(token.value, args)

            elif self.current_token.type == "LBRACKET":  # List indexing
                self.eat("LBRACKET")
                index = self.expr()
                self.eat("RBRACKET")
                return ListIndexNode(token.value, index)
            elif self.current_token.type == "DOT":  # Method call
                self.eat("DOT")
                method_name = self.current_token.value
                self.eat("IDENTIFIER")
                if self.current_token.type == "LPAREN":
                        self.eat("LPAREN")
                        args = []
                        if self.current_token.type != "RPAREN":
                            args.append(self.expr())
                            while self.current_token.type == "COMMA":
                                self.eat("COMMA")
                                args.append(self.expr())
                        self.eat("RPAREN")
                        # `id` was your owner expression
                        return MethodCall(owner, method_name, args)
                else:
                        # plain field access
                        return AttributeAccess(owner, method_name)
            else:
                # This is a regular variable reference
                var_node = VarNode(token.value)
                # Apply any prefixers to the variable reference
                if prefixers:
                    return TransformNode(':'.join(prefixers), var_node)
                return var_node

        elif token.type == "LPAREN":
            self.eat("LPAREN")
            node = self.expr()
            self.eat("RPAREN")
            return node

        elif token.type == "LBRACKET":  # List
            return self.list_statement()

        elif token.type == "LBRACE":  # Dictionary
            return self.dict_statement()

        else:
            self.error(f"Unexpected token: {token.type}")

        for prefixer in reversed(prefixers):
         value = TransformNode(prefixer, value)

        return value

    def term(self) -> AST:
        node = self.factor()
        while self.current_token.type in ("MUL", "DIV", "MOD", "AND", "XOR"):
            op = self.current_token
            if op.type == "MUL":
                self.eat("MUL")
            elif op.type == "MOD":
                self.eat("MOD")
            elif op.type == "DIV":
                self.eat("DIV")
            elif op.type == "XOR":
                self.eat("XOR")
            else:
                self.eat("AND")
            node = BinOp(left=node, op=op, right=self.factor())
        return node

    def expr(self) -> AST:
        node = self.term()
        while self.current_token.type in ("PLUS", "MINUS"):
            op = self.current_token
            if op.type == "PLUS":
                self.eat("PLUS")
            else:
                self.eat("MINUS")
            node = BinOp(left=node, op=op, right=self.term())
        return node

    def function_def(self) -> FunctionDef:
        self.eat("FUNC")
        if self.current_token.type != "IDENTIFIER":
            self.error("Expected function name after 'func'")
        func_name = self.current_token.value
        self.eat("IDENTIFIER")

        parameters = []
        # If a LPAREN is provided, parse parameters; otherwise assume none.
        if self.current_token.type == "LPAREN":
            self.eat("LPAREN")
            while self.current_token.type != "RPAREN":
                if self.current_token.type != "IDENTIFIER":
                    self.error("Expected parameter name")
                param_name = self.current_token.value
                self.eat("IDENTIFIER")

                # Check if there's a default value (IDENTIFIER ASSIGN expr)
                default_value = None
                if self.current_token.type == "ASSIGN":
                    self.eat("ASSIGN")
                    default_value = self.expr()  # Parse the default value as an expression

                # Add the parameter and its default value (if any) to the list
                parameters.append((param_name, default_value))

                # Handle COMMA for multiple parameters
                if self.current_token.type == "COMMA":
                    self.eat("COMMA")
                else:
                    break  # No more parameters
            self.eat("RPAREN")

        self.eat("LBRACE")
        body = []
        while self.current_token.type != "RBRACE":
            body.append(self.statement())
        self.eat("RBRACE")

        return FunctionDef(func_name, parameters, body)

    def return_statement(self) -> ReturnNode:
        """Parses a return statement: return expr"""
        self.eat("RETURN")
        expr_node = self.expr()
        return ReturnNode(expr_node)
    
    def class_def(self) -> ClassDef:
        self.eat("CLASS")
        class_name = self.current_token.value
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        body = []
        while self.current_token.type != "RBRACE":
            body.append(self.statement())
        self.eat("RBRACE")
        node = ClassDef(class_name, body)
        self.classes[class_name] = node  # Store the class for later lookups.
        return node
    
    def block(self):
        """Parses a block of statements enclosed in `{}`."""
        statements = []
        self.eat("LBRACE")  # Expect `{`
        
        while self.current_token.type != "RBRACE":
            statements.append(self.statement())

        self.eat("RBRACE")  # Expect `}`
        return statements
    
    def comparison(self):
        """Handles comparison expressions (e.g., x < 5, a == b)."""
        node = self.expr()  # Parse left-hand side

        while self.current_token.type in ("LT", "LE", "GT", "GE", "EQ", "NE"):
            op = self.current_token
            self.eat(op.type)
            right = self.expr()  # Parse right-hand side
            node = CompareNode(left=node, op=op, right=right)

        return node
    
    def peek(self) -> Token:
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return Token("EOF", None, self.pos)

    def statement(self) -> AST:
        if self.current_token.type == "PRINT":
            self.eat("PRINT")
            exprs = [self.expr()]
            while self.current_token.type == "COMMA":
                self.eat("COMMA")
                exprs.append(self.expr())
            return PrintNode(exprs)
        
        if self.current_token.type == "NONE":
            self.eat("NONE")
            return NoneNode()
        
        elif self.current_token.type == "IMPORT":
            return self.import_statement()
        
        elif self.current_token.type == "LOCAL":
            self.eat("LOCAL")
            var_name = self.current_token.value
            self.eat("IDENTIFIER")
            self.eat("ASSIGN")
            expr_node = self.expr()
            return LocalAssignNode(var_name, expr_node)
        
        elif self.current_token.type == "CONST":
            self.eat("CONST")
            var_name = self.current_token.value
            self.eat("IDENTIFIER")
            self.eat("ASSIGN")
            expr_node = self.expr()
            return ConstAssignNode(var_name, expr_node)

        elif self.current_token.type == "FUNC":
            return self.function_def()

        elif self.current_token.type == "RETURN":
            return self.return_statement()
        
        elif self.current_token.type == "CLASS":
            return self.class_def()


        elif self.current_token.type == "IDENTIFIER":
            token = self.current_token
            id = token.value
            self.eat("IDENTIFIER")

            # ——— ARROW: x -> expr ———
            if self.current_token.type == "ARROW":
                self.arrow_statement(id)

            # ——— DARROW: x -* ———
            if self.current_token.type == "DARROW":
                self.darrow_statment(id)


            # ——— method call: foo.bar(...) ———
            if self.current_token.type == "DOT":
                self.eat("DOT")
                method_name = self.current_token.value
                self.eat("IDENTIFIER")
                self.eat("LPAREN")
                args = []
                if self.current_token.type != "RPAREN":
                    args.append(self.expr())
                    while self.current_token.type == "COMMA":
                        self.eat("COMMA")
                        args.append(self.expr())
                self.eat("RPAREN")
                return MethodCall(token.value, method_name, args)

            # ——— function or constructor call: foo(...) ———
            if self.current_token.type == "LPAREN":
                self.eat("LPAREN")
                args = []
                if self.current_token.type != "RPAREN":
                    args.append(self.expr())
                    while self.current_token.type == "COMMA":
                        self.eat("COMMA")
                        args.append(self.expr())
                self.eat("RPAREN")
                if token.value in self.classes:
                    return ClassInstantiation(token.value, args)
                return FunctionCall(token.value, args)

            # ——— assignment: foo = … ———
            elif self.current_token.type == "ASSIGN":
                self.eat("ASSIGN")
                expr_node = self.expr()
                return AssignNode(token.value, expr_node)

            # ——— bare variable reference ———
            else:
                return VarNode(token.value)

        elif self.current_token.type == "WHILE":
            self.eat("WHILE")
            condition = self.comparison()  # Use comparison() for condition
            body = self.block()
            return WhileNode(condition, body)
        
        elif self.current_token.type == "FOR":
            self.eat("FOR")
            variable = self.current_token.value
            self.eat("IDENTIFIER")
            self.eat("IN")
            iterable = self.expr()
            body = self.block()
            return ForNode(variable, iterable, body)
            
        
        elif self.current_token.type == "IF":
            self.eat("IF")
            condition = self.comparison()
            self.eat("LBRACE")
            true_branch = []
            while self.current_token.type != "RBRACE":
                true_branch.append(self.statement())
            self.eat("RBRACE")
            elif_branches = []
            while self.current_token.type == "ELIF":
                self.eat("ELIF")
                elif_condition = self.comparison()
                self.eat("LBRACE")
                elif_body = []
                while self.current_token.type != "RBRACE":
                    elif_body.append(self.statement())
                self.eat("RBRACE")
                elif_branches.append((elif_condition, elif_body))
            false_branch = None
            if self.current_token.type == "ELSE":
                self.eat("ELSE")
                self.eat("LBRACE")
                false_branch = []
                while self.current_token.type != "RBRACE":
                    false_branch.append(self.statement())
                self.eat("RBRACE")
            return IfNode(condition, true_branch, elif_branches, false_branch)

        else:
            # Parse as an expression statement (e.g., method calls)
            return self.expr()


    def parse(self) -> typing.List[AST]:
        """Parse the token list and return the AST for all statements."""
        statements = []
        while self.current_token.type != "EOF":
            stmt = self.statement()
            statements.append(stmt)
        if self.debug:
         print(statements)
        return statements


# -------------------------------
#          Interpreter
# -------------------------------
class ReturnValue(Exception):
    """Custom exception to handle function return values."""
    def __init__(self, value: typing.Any):
        self.value = value


class Interpreter:
    def __init__(self):
        self.variables = {}  # Dictionary to store variable values
        self.functions = {}  # Dictionary to store functions
        self.classes = {}
        self.modules = {}
        self.native_builtins = {
            "math": self._import_math,
            "random": self._import_random,
            "string": self._import_string,
            "sys": self._import_sys,
            "os": self._import_os,
            "ctypes": self._import_ctypes,
            "IEngine": self._import_iengine
        }
        self.local_scope = None 
        self.constants = {}


    def _import_math(self):
        from math import (
        sin, cos, tan, asin, acos, atan,
        sqrt, log, log10, exp,
        floor, ceil, fabs, factorial,
        degrees, radians, pi, e, hypot, pow
    )

        self.variables["sin"] = lambda x: sin(x)
        self.variables["cos"] = lambda x: cos(x)
        self.variables["tan"] = lambda x: tan(x)
        self.variables["asin"] = lambda x: asin(x)
        self.variables["acos"] = lambda x: acos(x)
        self.variables["atan"] = lambda x: atan(x)

        self.variables["sqrt"] = lambda x: sqrt(x)
        self.variables["log"] = lambda x: log(x)
        self.variables["log10"] = lambda x: log10(x)
        self.variables["exp"] = lambda x: exp(x)

        self.variables["floor"] = lambda x: floor(x)
        self.variables["ceil"] = lambda x: ceil(x)
        self.variables["abs"] = lambda x: fabs(x)  # float abs

        self.variables["factorial"] = lambda x: factorial(x)
        self.variables["degrees"] = lambda x: degrees(x)
        self.variables["radians"] = lambda x: radians(x)

        self.variables["pi"] = pi
        self.variables["e"] = e

        self.variables['round'] = round
        self.variables["hypot"] = lambda x, y: hypot(x, y)
        self.variables["pow"] = lambda x, y: pow(x, y)

    def _import_random(self):
        from random import randint
        
        self.variables["rint"] = lambda a, b: randint(a, b)
        

    def _import_string(self):
        self.variables["isdigit"] = lambda s: s.isdigit()
        self.variables["isalpha"] = lambda s: s.isalpha()
        self.variables["isspace"] = lambda s: s.isspace()
        self.variables["replace"] = lambda s, old, new: s.replace(old, new)
        self.variables["startswith"] = lambda s, prefix: s.startswith(prefix)
        self.variables["endswith"] = lambda s, suffix: s.endswith(suffix)
        self.variables["split"] = lambda s, sep=None: s.split(sep)
        self.variables["join"] = lambda s, seq: s.join(seq)
        self.variables["strip"] = lambda s: s.strip()
        self.variables["lstrip"] = lambda s: s.lstrip()
        self.variables["rstrip"] = lambda s: s.rstrip()
        self.variables["str"] = lambda s: str(s)
        self.variables["getitem"] = lambda lst, index: lst[index] if isinstance(lst, list) and isinstance(index, int) else None
        self.variables["int"] = lambda x: int(x),
        self.variables['recurb'] = lambda n: list(range(n, 0, -1)),
        self.variables["lstartswith"] = lambda lst, prefix: (
        isinstance(lst, list) and 
        len(lst) > 0 and 
        isinstance(lst[0], str) and 
        lst[0].startswith(prefix)
        )
        
    def _import_sys(self):
        self.variables["sys.argv"] = sys.argv
        self.variables["sys~exit"] = sys.exit
        self.variables["sys~version"] = "ios 4.6"
        self.variables["sys~platform"] = sys.platform
        self.variables["sys~getsizeof"] = sys.getsizeof
        self.variables["sys~getfilesystemencoding"] = sys.getfilesystemencoding

    def _import_os(self):
        self.variables["os~getcwd"] = os.getcwd
        self.variables["os~chdir"] = os.chdir
        self.variables["os~listdir"] = os.listdir
        self.variables["os~remove"] = os.remove
        self.variables["os~system"] = os.system 

    def _import_ctypes(self):
        self.variables["lcdll"] = lambda dll_path: ctypes.cdll.LoadLibrary(dll_path)
        self.variables["c_int"] = ctypes.c_int
        self.variables["c_char_p"] = ctypes.c_char_p
        self.variables["c_double"] = ctypes.c_double
        self.variables["c_float"] = ctypes.c_float  # Added c_float
        self.variables["c_long"] = ctypes.c_long    # Added c_long
        self.variables["c_ulong"] = ctypes.c_ulong  # Added c_ulong
        self.variables["c_short"] = ctypes.c_short  # Added c_short
        self.variables["c_ushort"] = ctypes.c_ushort  # Added c_ushort
        self.variables["c_bool"] = ctypes.c_bool    # Added c_bool
        self.variables["c_wchar"]    = ctypes.c_wchar        # single wide char
        self.variables["c_wchar_p"]  = ctypes.c_wchar_p      # wide‐char string
        self.variables["settypes"] =  lambda s, t, types: setattr(getattr(s, t), 'argtypes', types)
        self.variables["setrestype"] =  lambda s, t, restype: setattr(getattr(s, t), 'restype', restype)
        self.variables["calld"] = lambda dll, func, *args: getattr(dll, func)(
    *[arg.encode() if isinstance(arg, str) else arg for arg in args])
        self.variables["byref"] = ctypes.byref   # Added byref method
        self.variables["sizeof"] = ctypes.sizeof  # Added sizeof method
        self.variables["pointer"] = ctypes.POINTER    # To create pointers for types
        self.variables["cast_pointer"] = lambda type_, value: ctypes.cast(value, ctypes.POINTER(type_))  # To cast to pointer type
        self.variables["string_at"] = ctypes.string_at  # To read string from memory address (useful for DLLs returning string pointers)
        self.variables["memmove"] = ctypes.memmove    # To move memory from one location to another
        self.variables["create_buffer"] = lambda size: ctypes.create_string_buffer(size)  # Create a mutable buffer of bytes
        self.variables["addressof"] = ctypes.addressof  # Get the address of a ctypes object
        self.variables["wchar_t"] = ctypes.wchar_t      # For wide characters
        self.variables["c_uint8"] = ctypes.c_uint8      # For unsigned 8-bit integer
        self.variables["c_int8"] = ctypes.c_int8        # For signed 8-bit integer
        self.variables["c_int_p"] = ctypes.POINTER(ctypes.c_int)  # For pointer to c_int
        self.variables["c_char_p_p"] = ctypes.POINTER(ctypes.c_char_p)  # Pointer to c_char_p
    
    def _import_iengine(self):
        self.variables["IEngine"] = Ursina
        self.variables["run"] = lambda x: x.run()
        self.variables["Entity"] = lambda m, c, s, col: Entity(model=m, color=c, scale=s, collider=col),
        self.variables["hsv"] = lambda r, g, b: hsv(r,g,b)



    def visit(self, node):

        if isinstance(node, PrintNode):
            values = []
            for expr in node.exprs:
                val = self.visit(expr)

                # Check for __repr__ method first
                if isinstance(val, dict) and "__repr__" in val:
                    repr_func = val["__repr__"]
                    old_vars = self.variables.copy()
                    self.variables = val.copy()
                    try:
                        for stmt in repr_func.body:
                            self.visit(stmt)
                    except ReturnValue as rv:
                        val = rv.value
                    self.variables = old_vars
                # Fallback to __str__ if __repr__ is not present
                elif isinstance(val, dict) and "__str__" in val:
                    str_func = val["__str__"]
                    old_vars = self.variables.copy()
                    self.variables = val.copy()
                    try:
                        for stmt in str_func.body:
                            self.visit(stmt)
                    except ReturnValue as rv:
                        val = rv.value
                    self.variables = old_vars

                values.append(str(val))


            
            os.write(1, (' '.join(values) + '\n').encode())

        
        elif isinstance(node, ClassDef):
           self.classes[node.class_name] = node

        elif isinstance(node, ClassInstantiation):
            # Retrieve class definition
            class_def = self.classes.get(node.class_name)
            if not class_def:
                raise NameError(f"Unknown class '{node.class_name}'")
            
            # Create an instance dictionary with the class name
            instance = {"__class__": node.class_name}
            
            # Lookup the __init__ method
            init_method = next((stmt for stmt in class_def.body if isinstance(stmt, FunctionDef) and stmt.func_name == "__init__"), None)
            if init_method:
                # Evaluate arguments for __init__
                args = [self.visit(arg) for arg in node.args]
                
                # Save current scope, and use instance as environment for fields
                old_vars = self.variables.copy()
                self.variables = instance  # Set instance variables as the scope
                
                # Assign parameters to instance (making them available in all methods)
                for param, arg_val in zip(init_method.parameters, args):
                    self.variables[param] = arg_val
                
                try:
                    # Execute the body of the __init__ method
                    for stmt in init_method.body:
                        self.visit(stmt)
                except ReturnValue as rv:
                    pass
                
                # Update instance after constructor execution
                instance.update(self.variables)  # Apply instance fields (parameters)
                self.variables = old_vars
            
            # Assign methods to the instance, binding the instance as the first argument
            for stmt in class_def.body:
                if isinstance(stmt, FunctionDef):
                    def method_func(args, stmt=stmt):  # Capture stmt correctly
                        if len(args) != len(stmt.parameters) + 1:  # Include instance as first argument
                            raise TypeError(f"Method '{stmt.func_name}' expects {len(stmt.parameters) + 1} arguments, got {len(args)}")
                        
                        # Pass instance explicitly as the first argument
                        args = [instance] + args  # The instance is the first argument

                        # Save current scope and inject 'self' (the instance)
                        old_vars = self.variables.copy()
                        self.variables = instance.copy()  # Use the instance's scope
                        
                        for param, arg_val in zip(stmt.parameters, args[1:]):  # Skip the first argument
                            self.variables[param] = arg_val
                        
                        try:
                            for method_stmt in stmt.body:
                                self.visit(method_stmt)
                        except ReturnValue as rv:
                            self.variables = old_vars
                            return rv.value
                        
                        self.variables = old_vars
                    
                    # Add the method to the instance, with the method bound to 'self'
                    instance[stmt.func_name] = method_func
            
            return instance



        elif isinstance(node, ImportNode):

            # 1) built-in?
            mod = node.module_name
            if mod in self.native_builtins:
                self.native_builtins[mod]()  # calls _import_math(), etc.
                self.modules[mod] = True     # mark as loaded
                return

            # 2) filesystem: foo.bar → foo/bar.ios
            rel = mod.replace(".", os.sep) + ".ios"
            for d in [".", "lib", "std"]:
                path = os.path.join(d, rel)
                if os.path.isfile(path):
                    source = open(path, encoding="utf-8").read()
                    tree   = Parser(Lexer(source).tokenize()).parse()
                    exports = {}
                    for stmt in tree:
                        if isinstance(stmt, FunctionDef):
                            self.functions[stmt.func_name] = stmt
                            exports[stmt.func_name] = stmt
                        elif isinstance(stmt, ClassDef):
                            self.classes[stmt.class_name] = stmt
                            exports[stmt.class_name] = stmt
                        elif isinstance(stmt, AssignNode):
                            val = self.visit(stmt.expr)
                            self.variables[stmt.var_name] = val
                            exports[stmt.var_name] = val
                    self.modules[mod] = exports
                    return

            # 3) not found
            raise ImportError(f"Module '{mod}' not found in built-ins or on disk")
        

        elif isinstance(node, LocalAssignNode):
            value = self.visit(node.expr)
            if self.local_scope is not None:
                self.local_scope[node.var_name] = value
            else:
                raise RuntimeError("Local assignment outside of function")
            
        elif isinstance(node, ConstAssignNode):
            if node.var_name in self.constants:
                raise RuntimeError(f"Constant '{node.var_name}' already defined")
            value = self.visit(node.expr)
            self.constants[node.var_name] = value

        elif isinstance(node, AssignNode):
            value = self.visit(node.expr)
            if node.var_name in self.constants:
                raise RuntimeError(f"Cannot reassign constant '{node.var_name}'")
            if "." in node.var_name:  # Handle field assignment (e.g., main.__type__ = 'str')
                obj_name, field_name = node.var_name.split(".", 1)
                if obj_name in self.variables and isinstance(self.variables[obj_name], dict):
                    self.variables[obj_name][field_name] = value
                else:
                    raise AttributeError(f"Object '{obj_name}' has no attribute '{field_name}'")
            else:
                if self.local_scope is not None and node.var_name in self.local_scope:
                    self.local_scope[node.var_name] = value
                else:
                    self.variables[node.var_name] = value

        elif isinstance(node, VarNode):
             if self.local_scope is not None and node.var_name in self.local_scope:
                return self.local_scope[node.var_name]
             elif node.var_name in self.variables:
                return self.variables[node.var_name]
             elif node.var_name in self.constants:
                return self.constants[node.var_name]
             else:
                raise NameError(f"{Fore.RED}Error:{Style.RESET_ALL} Undefined variable '{Fore.YELLOW}{node.var_name}{Style.RESET_ALL}'")

        elif isinstance(node, BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)

            # Handle the PLUS operator directly
            if node.op.type == "PLUS":
                return left + right

            elif node.op.type == "MINUS":
                return left - right
            elif node.op.type == "MUL":
                return left * right
            elif node.op.type == "DIV":
                return left / right
            elif node.op.type == "MOD":
                return left % right
            elif node.op.type == "AND":
                return left & right
            elif node.op.type == "XOR":
                return left ^ right


        elif isinstance(node, Num):
            return node.value

        elif isinstance(node, Str):
            def vreplace(match):
                    var_name = match.group(1)
                    if var_name in self.variables:
                        return str(self.variables[var_name])
                    else:
                        raise NameError(f"Undefined variable '{var_name}' in cmdin string")
                    
            node.value = re.sub(r'\{(\w+)\}', vreplace, node.value)
            return node.value

        elif isinstance(node, Bool):
            return node.value

        elif isinstance(node, FunctionDef):
            self.functions[node.func_name] = node

        elif isinstance(node, FunctionCall):
            func_name = node.func_name

            # ✅ 1. Native built-in Python function
            if func_name in self.variables and callable(self.variables[func_name]):
                args = [self.visit(arg) for arg in node.args]
                return self.variables[func_name](*args)

            # ✅ 2. User-defined IOS function
            if func_name not in self.functions:
                raise NameError(f"Undefined function '{func_name}'")

            func_def = self.functions[func_name]
            args = [self.visit(arg) for arg in node.args]

            if len(args) != len(func_def.parameters):
                raise TypeError(f"Function '{func_name}' expected {len(func_def.parameters)} arguments but got {len(args)}")

            # 🌍 Keep a copy of the global scope
            global_vars = self.variables

            previous_local_scope = self.local_scope
            self.local_scope = {}

            # Set function parameters in global scope (default behavior)
            for param_tuple, value in zip(func_def.parameters, args):
                param_name = param_tuple[0]
                self.local_scope[param_name] = value

            try:
                for stmt in func_def.body:
                    self.visit(stmt)
                ret_value = None
            except ReturnValue as rv:
                ret_value = rv.value

            self.local_scope = previous_local_scope

            return ret_value




        elif isinstance(node, ReturnNode):
            value = self.visit(node.expr)
            raise ReturnValue(value)
        
        elif isinstance(node, CompareNode):
                left = self.visit(node.left)
                right = self.visit(node.right)

                if node.op.type == "LT":
                    return left < right
                elif node.op.type == "LE":
                    return left <= right
                elif node.op.type == "GT":
                    return left > right
                elif node.op.type == "GE":
                    return left >= right
                elif node.op.type == "EQ":
                    return left == right
                elif node.op.type == "NE":
                    return left != right
                else:
                    raise Exception(f"Unknown comparison operator {node.op.type}")

        elif isinstance(node, WhileNode):
                while self.visit(node.condition):  # Now condition supports <, >, etc.
                    for stmt in node.body:
                        self.visit(stmt)

        elif isinstance(node, IfNode):
            # Evaluate the condition and cache it
            condition_result = self.visit(node.condition)  
            
            # Handle if the condition is true
            if condition_result:
                for stmt in node.true_branch:
                    self.visit(stmt)
            else:
                # Iterate through elif branches and check conditions only once
                for elif_condition, elif_body in node.elif_branches:
                    elif_result = self.visit(elif_condition)
                    if elif_result:  # If the elif condition is true, process the body
                        for stmt in elif_body:
                            self.visit(stmt)
                        break  # Stop processing once the first true branch is found
                else:
                    # If no condition is true, process the else branch (if it exists)
                    if node.false_branch:
                        for stmt in node.false_branch:
                            self.visit(stmt)

        elif isinstance(node, ForNode):
            iterable = self.visit(node.iterable)
            
            if not hasattr(iterable, '__iter__'):
                raise TypeError("For loop expects an iterable (like list, string, dict, tuple, or set)")
            
            # If it's a dictionary, iterate over keys by default (like Python does)
            if isinstance(iterable, dict):
                loop_items = iterable.values()
            else:
                loop_items = iterable

            for item in loop_items:
                self.variables[node.variable] = item
                for stmt in node.body:
                    self.visit(stmt)

        elif isinstance(node, NoneNode):
            return None
        
        elif isinstance(node, List):
            # Visit each element in the list and return the evaluated list
            return [self.visit(element) for element in node.elements]
        
        elif isinstance(node, ListIndexNode):
            if node.list_name not in self.variables:
                raise NameError(f"Undefined variable '{node.list_name}'")
            container = self.variables[node.list_name]
            index = self.visit(node.index)
            if isinstance(container, list):
                if not isinstance(index, int):
                    raise TypeError(f"List index must be an integer, got {type(index).__name__}")
                try:
                    return container[index]
                except IndexError:
                    raise IndexError(f"Index {index} out of range for list '{node.list_name}'")
            elif isinstance(container, dict):
                try:
                    return container[index]
                except KeyError:
                    raise KeyError(f"Key {index} not found in dict '{node.list_name}'")
            else:
                raise TypeError(f"Variable '{node.list_name}' is not indexable")
            
        elif isinstance(node, MethodCall):
            obj = self.variables[node.list_name]
            if isinstance(obj, dict) and "__class__" in obj:
                class_def = self.classes[obj["__class__"]]
                for stmt in class_def.body:
                    if isinstance(stmt, FunctionDef) and stmt.func_name == node.method_name:
                        old_vars = self.variables.copy()
                        # Set up the method scope with the instance fields.
                        self.variables = obj.copy()
                        args = [self.visit(arg) for arg in node.args]
                        if len(args) != len(stmt.parameters):
                            raise TypeError("Method argument count mismatch")
                        for param, arg_val in zip(stmt.parameters, args):
                            self.variables[param] = arg_val
                        try:
                            for s in stmt.body:
                                self.visit(s)
                        except ReturnValue as rv:
                            ret_val = rv.value
                        else:
                            ret_val = None
                        # Restore variable scope
                        self.variables = old_vars
                        return ret_val
            # fall back to previous MethodCall handling (e.g. for lists/dicts)

            # Get the list or dictionary by its name
            if node.list_name not in self.variables:
                raise NameError(f"Undefined variable '{node.list_name}'")
            container = self.variables[node.list_name]

            # Evaluate the arguments
            args = [self.visit(arg) for arg in node.args]

            # Handle list methods
            if isinstance(container, list):
                if node.method_name == "append":
                    if len(args) != 1:
                        raise TypeError("append() expects exactly 1 argument")
                    container.append(args[0])
                elif node.method_name == "pop":
                    if len(args) > 1:
                        raise TypeError("pop() expects at most 1 argument")
                    index = args[0] if args else -1
                    if not isinstance(index, int):
                        raise TypeError("pop() expects an integer argument")
                    try:
                        return container.pop(index)
                    except IndexError:
                        raise IndexError(f"Index {index} out of range for list '{node.list_name}'")
                elif node.method_name == "remove":
                    if len(args) != 1:
                        raise TypeError("remove() expects exactly 1 argument")
                    try:
                        container.remove(args[0])
                    except ValueError:
                        raise ValueError(f"Value {args[0]} not found in list '{node.list_name}'")
                elif node.method_name == "clear":
                    if len(args) != 0:
                        raise TypeError("clear() expects no arguments")
                    container.clear()
                elif node.method_name == "insert":
                    if len(args) != 2:
                        raise TypeError("insert() expects exactly 2 arguments")
                    index, value = args
                    if not isinstance(index, int):
                        raise TypeError("insert() expects the first argument to be an integer")
                    container.insert(index, value)
                elif node.method_name == "reverse":
                    if len(args) != 0:
                        raise TypeError("reverse() expects no arguments")
                    container.reverse()
                elif node.method_name == "sort":
                    if len(args) != 0:
                        raise TypeError("sort() expects no arguments")
                    container.sort()
                else:
                    raise AttributeError(f"List has no method '{node.method_name}'")

            # Handle dictionary methods
            elif isinstance(container, dict):
                if node.method_name == "get":
                    if len(args) not in (1, 2):
                        raise TypeError("get() expects 1 or 2 arguments")
                    key = args[0]
                    default = args[1] if len(args) == 2 else None
                    return container.get(key, default)
                elif node.method_name == "keys":
                    if len(args) != 0:
                        raise TypeError("keys() expects no arguments")
                    return list(container.keys())
                elif node.method_name == "values":
                    if len(args) != 0:
                        raise TypeError("values() expects no arguments")
                    return list(container.values())
                elif node.method_name == "items":
                    if len(args) != 0:
                        raise TypeError("items() expects no arguments")
                    return list(container.items())
                elif node.method_name == "pop":
                    if len(args) != 1:
                        raise TypeError("pop() expects exactly 1 argument")
                    key = args[0]
                    try:
                        return container.pop(key)
                    except KeyError:
                        raise KeyError(f"Key {key} not found in dictionary '{node.list_name}'")
                elif node.method_name == "add":
                    if len(args) != 1 or not isinstance(args[0], dict):
                        raise TypeError("update() expects exactly 1 dictionary argument")
                    container.update(args[0])
                elif node.method_name == "clear":
                    if len(args) != 0:
                        raise TypeError("clear() expects no arguments")
                    container.clear()

                else:
                    raise AttributeError(f"Dictionary has no method '{node.method_name}'")

            else:
                raise TypeError(f"Variable '{node.list_name}' is not a list or dictionary")

        elif isinstance(node, Dict):
            # Evaluate each key-value pair and return a dictionary
            return {self.visit(key): self.visit(value) for key, value in node.elements}
        
        elif isinstance(node, TransformNode):
            return self.visit_TransformNode(node)
        
        elif isinstance(node, ArrowNote):
             self.variables[node.var_name] = node.cvalue.value

        elif isinstance(node, DArrowNote):
            del self.variables[node.var_name]

        elif isinstance(node, AttributeAccess):
            # evaluate the owner expression, which should give you the instance dict
            inst = self.visit(node.owner)
            return inst[node.name]

        else:
            raise Exception(f"Unknown node type: {type(node)}")
        
    def visit_TransformNode(self, node: TransformNode):
        # 1) Evaluate the inner expression (could be Str, VarNode, etc.)
        value = self.visit(node.value)

        # 2) If it’s a variable name, lookup its runtime value
        if isinstance(value, str) and value in self.variables:
            value = self.variables[value]
        # 3) If it’s a Str AST, extract the .value
        elif hasattr(node.value, 'value'):
            value = node.value.value

        # 4) Apply each transform in sequence
        for spec in node.transform.split(':'):
            spec = spec.strip()
            # Parametric transform e.g. rep(b,a) or rp(x,y,2)
            if '(' in spec and spec.endswith(')'):
                name, args_str = spec.split('(', 1)
                args_str = args_str[:-1]  # remove trailing ')'
                args = [arg.strip() for arg in args_str.split(',')] if args_str else []

                if name not in extended_transforms:
                    raise RuntimeError(f"Unknown transform '{name}'")
                func = extended_transforms[name]
                
                # Coerce argument types if needed
                def _parse_arg(a: str):
                    if a.isdigit():
                        return int(a)
                    if (a.startswith("'") and a.endswith("'")) or (a.startswith('"') and a.endswith('"')):
                        return a[1:-1]
                    return a
                
                parsed_args = [_parse_arg(a) for a in args]
                value = func(value, *parsed_args)

            else:
                # Simple transform: 'u', 'r', 'ast', etc.
                name = spec
                if name not in extended_transforms:
                    raise RuntimeError(f"Unknown transform '{name}'")
                value = extended_transforms[name](value)
       
        return value if value is not None else None


    def interpret(self, tree: typing.List[AST]) -> None:
        """Interpret each statement in the AST."""
        for stmt in tree:
            self.visit(stmt)

def preprocess(source: str) -> str:
    # Step 1: Extract @define macros (now supporting any characters, not just \w+)
    define_pattern = r'@define\s+(.+?)(?:\((.*?)\))?\s+(.+)'
    raw_defines = re.findall(define_pattern, source)

    # Step 2: Build replacements dict
    replacements = {}
    for name, args, replacement in raw_defines:
        name = name.strip()
        replacements[name] = {
            'args': [arg.strip() for arg in args.split(',')] if args else None,
            'replacement': replacement.strip()
        }

    # Step 3: Remove @define lines from source
    source = re.sub(r'^@define .+', '', source, flags=re.MULTILINE)

    # Step 4: Resolve chained/nested defines
    def resolve_nested():
        changed = True
        while changed:
            changed = False
            for key, val in replacements.items():
                rep = val['replacement']
                if not val['args']:  # only simple replacements
                    if rep in replacements and not replacements[rep]['args']:
                        replacements[key]['replacement'] = replacements[rep]['replacement']
                        changed = True

    resolve_nested()

    # Step 5: Apply all macros to the source
    def apply_macros(src: str) -> str:
        # Apply function-like macros first
        for name, data in replacements.items():
            if data['args']:
                # Escape special characters in name
                escaped_name = re.escape(name)
                pattern = rf'{escaped_name}\((.*?)\)'
                def repl(m):
                    passed_args = [a.strip() for a in m.group(1).split(',')]
                    if len(passed_args) != len(data['args']):
                        return m.group(0)
                    result = data['replacement']
                    for arg_name, val in zip(data['args'], passed_args):
                        result = result.replace(arg_name, val)
                    return result
                src = re.sub(pattern, repl, src)

        # Apply simple defines
        for name, data in sorted(replacements.items(), key=lambda x: -len(x[0])):
            if not data['args']:
                escaped_name = re.escape(name)
                src = re.sub(escaped_name, data['replacement'], src)

        return src

    source = apply_macros(source)

    # Step 6: Handle @repeat blocks
    def handle_repeat(m):
        count = int(m.group(1)) + 1
        block = m.group(2)
        return (block.strip() + '\n') * count

    source = re.sub(r'@repeat (\d+):\s*(.+)', handle_repeat, source)

    # Step 7: Handle #ifdef / #ifndef blocks
    def handle_conditionals(src: str) -> str:
        def ifdef_block(m):
            macro = m.group(1)
            code = m.group(2)
            return code if macro in replacements else ''

        def ifndef_block(m):
            macro = m.group(1)
            code = m.group(2)
            return code if macro not in replacements else ''

        src = re.sub(r'#ifdef (.+?)\n(.*?)#endif', ifdef_block, src, flags=re.DOTALL)
        src = re.sub(r'#ifndef (.+?)\n(.*?)#endif', ifndef_block, src, flags=re.DOTALL)
        return src

    return handle_conditionals(source)
# -------------------------------
#              Main
# -------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python parser.py <filename>")
        return

    try:
        with open(sys.argv[1], "r", encoding="utf-8") as file:
            source = preprocess(file.read())
        
        # Tokenization
        lexer = Lexer(source, debug=debug)
        tokens = lexer.tokenize()

        # Parsing
        parser = Parser(tokens, debug=debug)
        tree = parser.parse()

        # Interpretation
        interpreter = Interpreter()
        interpreter.interpret(tree)

        def pretty_ast(node, indent=0):
            prefix = " " * (indent * 4)  # Consistent 4-space indentation for each level
            parts = []

            if isinstance(node, AssignNode):
                parts.append(f"{prefix}Assign(")
                parts.append(f"{prefix}    var={repr(node.var_name)},")
                parts.append(f"{prefix}    value={pretty_ast(node.expr)}")
                parts.append(f"{prefix})")
            
            elif isinstance(node, PrintNode):
                exprs_str = ", ".join(pretty_ast(expr) for expr in node.exprs)
                parts.append(f"{prefix}Print(")
                parts.append(f"{prefix}    exprs=[")
                parts.append(exprs_str)
                parts.append(f"{prefix}    ]")
                parts.append(f"{prefix})")
            
            elif isinstance(node, BinOp):
                parts.append(f"{prefix}BinOp(")
                parts.append(f"{prefix}    left={pretty_ast(node.left, indent + 1)},")
                parts.append(f"{prefix}    op='{node.op.type}',")
                parts.append(f"{prefix}    right={pretty_ast(node.right, indent + 1)}")
                parts.append(f"{prefix})")
            
            elif isinstance(node, Num):
                parts.append(f"{prefix}Num(value={node.value})")
            
            elif isinstance(node, Str):
                parts.append(f"{prefix}Str(value={repr(node.value)})")
            
            elif isinstance(node, Bool):
                parts.append(f"{prefix}Bool(value={node.value})")
            
            elif isinstance(node, VarNode):
                parts.append(f"{prefix}Var(name={repr(node.var_name)})")
            
            elif isinstance(node, FunctionDef):
                body_str = "\n".join(pretty_ast(stmt, indent + 1) for stmt in node.body)
                parts.append(f"{prefix}FunctionDef(name={node.func_name}, params={node.parameters}, body=[")
                parts.append(body_str)
                parts.append(f"{prefix}])")
            
            elif isinstance(node, FunctionCall):
                args_str = ", ".join(pretty_ast(arg, indent + 1) for arg in node.args)
                parts.append(f"{prefix}FunctionCall(name={node.func_name}, args=[{args_str}])")
            
            elif isinstance(node, ReturnNode):
                parts.append(f"{prefix}Return({pretty_ast(node.expr, indent - 2)})")
            
            elif isinstance(node, CompareNode):
                parts.append(f"{prefix}Compare(")
                parts.append(f"{prefix}    left={pretty_ast(node.left, indent + 1)},")
                parts.append(f"{prefix}    op='{node.op.type}',")
                parts.append(f"{prefix}    right={pretty_ast(node.right, indent + 1)}")
                parts.append(f"{prefix})")
            
            elif isinstance(node, WhileNode):
                body_str = "\n".join(pretty_ast(stmt, indent + 1) for stmt in node.body)
                parts.append(f"{prefix}While(")
                parts.append(f"{prefix}    condition={pretty_ast(node.condition, indent + 1)},")
                parts.append(f"{prefix}    body=[")
                parts.append(body_str)
                parts.append(f"{prefix}    ]")
                parts.append(f"{prefix})")
            
            elif isinstance(node, IfNode):
                true_branch_str = "\n".join(pretty_ast(stmt, indent + 1) for stmt in node.true_branch)
                false_branch_str = ("\n".join(pretty_ast(stmt, indent + 1) for stmt in node.false_branch)
                                    if node.false_branch else "None")
                elif_branches_str = ""
                if node.elif_branches:
                    elif_branches_str = "\n".join(
                        f"{prefix}    Elif(condition={pretty_ast(cond, indent + 2)}, body=[\n" +
                        "\n".join(pretty_ast(s, indent + 3) for s in body) +
                        f"\n{prefix}    ])"
                        for cond, body in node.elif_branches)
                parts.append(f"{prefix}If(")
                parts.append(f"{prefix}    condition={pretty_ast(node.condition, indent + 1)},")
                parts.append(f"{prefix}    true_branch=[")
                parts.append(true_branch_str)
                parts.append(f"{prefix}    ],")
                parts.append(f"{prefix}    elif_branches=[")
                parts.append(elif_branches_str)
                parts.append(f"{prefix}    ],")
                parts.append(f"{prefix}    false_branch=[")
                parts.append(false_branch_str)
                parts.append(f"{prefix}    ]")
                parts.append(f"{prefix})")
            
            elif isinstance(node, List):
                elems_str = ", ".join(pretty_ast(elem, indent + 1) for elem in node.elements)
                parts.append(f"{prefix}List([{elems_str}])")
            
            elif isinstance(node, ListIndexNode):
                parts.append(f"{prefix}ListIndex(")
                parts.append(f"{prefix}    list_name={repr(node.list_name)},")
                parts.append(f"{prefix}    index={pretty_ast(node.index, indent + 1)}")
                parts.append(f"{prefix})")
            
            elif isinstance(node, Dict):
                pairs_str = ",\n".join(
                    f"{prefix}    ({pretty_ast(key, indent + 1)}, {pretty_ast(value, indent + 1)})"
                    for key, value in node.elements)
                parts.append(f"{prefix}Dict([")
                parts.append(pairs_str)
                parts.append(f"{prefix}])")
            
            elif hasattr(node, 'method_name'):  # For MethodCall node
                args_str = ", ".join(pretty_ast(arg, indent + 1) for arg in node.args)
                parts.append(f"{prefix}MethodCall(")
                parts.append(f"{prefix}    object={repr(node.list_name)},")
                parts.append(f"{prefix}    method='{node.method_name}',")
                parts.append(f"{prefix}    args=[{args_str}]")
                parts.append(f"{prefix})")
            
            elif isinstance(node, ImportNode):
                parts.append(f"{prefix}Import(module_name={repr(node.module_name)})")
            
            elif isinstance(node, NoneNode):
                parts.append(f"{prefix}NoneNode()")
            
            else:
                parts.append(f"{prefix}{repr(node)}")

            # Return the concatenated string
            return "\n".join(parts)

            
        async def presentation(tokens, tree, source_lines):
            iosnb_filename = f"{sys.argv[1].split('.')[0]}.iosnb"

            async with aiofiles.open(iosnb_filename, 'w', encoding="utf-8") as file:
                # Collect tokens and AST data into a list
                output_lines = [f"{tokens}\n\n"]
                output_lines += [pretty_ast(stmt) + "\n" for stmt in tree]

                # Write original source and AST info
                await file.writelines(source_lines)
                await file.write('\n\n')
                await file.writelines(output_lines)

        asyncio.run(presentation(tokens, tree, source))

    except Exception as e:
        print(f"{Fore.RED}Error:{Style.RESET_ALL} {e}")

if __name__ == "__main__":
    main()
