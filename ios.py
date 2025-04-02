import json
import sys
import os
from typing import Any, List, Optional
from colorama import Fore, Style, init

init(autoreset=True)


# -------------------------------
#          Token Definition
# -------------------------------
class Token:
    """Represents a token produced by the lexer."""
    def __init__(self, type_: str, value: Any = None, pos: Optional[int] = None) -> None:
        self.type = type_
        self.value = value
        self.pos = pos  # Position in the source (for error reporting)

    def __repr__(self) -> str:
        if self.value is not None:
            return f"Token({self.type}, {repr(self.value)})"
        return f"Token({self.type})"


# -------------------------------
#             Lexer
# -------------------------------
class Lexer:
    """Converts an input string into a stream of tokens."""
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.text else None

    def advance(self) -> None:
        """Advance the pointer and update the current character."""
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def skip_whitespace(self) -> None:
        """Skip any whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self) -> Token:
        """Return a NUMBER token from a sequence of digits."""
        result = ""
        start_pos = self.pos
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return Token("NUMBER", int(result), start_pos)

    def string(self) -> Token:
        """Return a STRING token by consuming characters inside single quotes."""
        start_pos = self.pos
        self.advance()  # Skip the opening quote
        result = ""
        while self.current_char is not None and self.current_char != "'":
            result += self.current_char
            self.advance()
        if self.current_char != "'":
            self.error("Unterminated string literal")
        self.advance()  # Skip the closing quote
        return Token("STRING", result, start_pos)

    def identifier(self) -> Token:
        """Handles variable names and reserved keywords."""
        result = ""
        start_pos = self.pos
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == "_"):
            result += self.current_char
            self.advance()
        # Reserved keywords
        if result == "print":
            return Token("PRINT", None, start_pos)
        elif result == "True":
            return Token("BOOL", True, start_pos)
        elif result == "False":
            return Token("BOOL", False, start_pos)
        elif result == "func":
            return Token("FUNC", None, start_pos)
        elif result == "return":
            return Token("RETURN", None, start_pos)
        elif result == "while":
         return Token("WHILE", None, start_pos)
        elif result == "if":
         return Token("IF", None, start_pos)
        elif result == "else":
         return Token("ELSE", None, start_pos)
        elif result == "import":
            return Token("IMPORT", None, start_pos)
        
        # Otherwise, it's a variable name (IDENTIFIER)
        return Token("IDENTIFIER", result, start_pos)
    
    def python_block(self):
        """Tokenizes a block of Python code as a single STRING token."""
        start_pos = self.pos
        self.eat("PYTHON")  # Consume the "python" keyword
        self.eat("LBRACE")  # Consume the opening brace

        python_code = ""
        brace_count = 1  # Keep track of nested braces
        while self.current_char is not None:
            python_code += self.current_char
            if self.current_char == '{':
                brace_count += 1
            elif self.current_char == '}':
                brace_count -= 1
                if brace_count == 0:  # Found the closing brace
                    self.advance()
                    break
            self.advance()

        if brace_count != 0:
            self.error("Unbalanced braces in Python block")

        return Token("PYTHON_BLOCK", python_code, start_pos)  # New token type

    def error(self, message: str) -> None:
        """Raise a syntax error with context information."""
        line_number = self.text.count('\n', 0, self.pos) + 1
        col = self.pos - self.text.rfind('\n', 0, self.pos)
        raise SyntaxError(f"{Fore.RED}{Style.BRIGHT}SyntaxError:{Style.RESET_ALL} "
                          f"{message} at line {line_number}, column {col}")

    def get_next_token(self) -> Token:
        """Lex and return the next token in the input text."""
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == "'":
                return self.string()

            if self.current_char.isalpha():
                return self.identifier()

            if self.current_char == '+':
                tok = Token("PLUS", None, self.pos)
                self.advance()
                return tok
            if self.current_char == '-':
                tok = Token("MINUS", None, self.pos)
                self.advance()
                return tok
            if self.current_char == '*':
                tok = Token("MUL", None, self.pos)
                self.advance()
                return tok
            if self.current_char == '/':
                tok = Token("DIV", None, self.pos)
                self.advance()
                return tok
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
            
            if self.current_char == '<':
                self.advance()
                if self.current_char == '=':  # Handle '<='
                    self.advance()
                    return Token("LE", None, self.pos)
                return Token("LT", None, self.pos)
            
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
            
            elif self.current_char == 'p':  # Check for "python"
                if self.text[self.pos:self.pos + 6] == "python":
                    return self.python_block()  # Call the new python_block method
                else:
                    return self.identifier()  # It's a regular identifier
                
            self.error(f"Unexpected character '{self.current_char}'")

        return Token("EOF", None, self.pos)

    def tokenize(self) -> List[Token]:
        """Tokenize the entire input text."""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == "EOF":
                break
        return tokens


# -------------------------------
#           AST Nodes
# -------------------------------
class AST:
    """Base class for all AST nodes."""
    pass


class PrintNode(AST):
    def __init__(self, exprs: List[AST]) -> None:
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
    def __init__(self, func_name: str, parameters: List[str], body: List[AST]) -> None:
        self.func_name = func_name
        self.parameters = parameters
        self.body = body

    def __repr__(self) -> str:
        return f"FunctionDef({self.func_name}, params={self.parameters}, body={self.body})"


class FunctionCall(AST):
    """Represents a function call."""
    def __init__(self, func_name: str, args: List[AST]) -> None:
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
    def __init__(self, condition: AST, body: List[AST]) -> None:
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
    """Represents an if-else statement."""
    def __init__(self, condition, true_branch, false_branch=None):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __repr__(self):
        return f"IfNode(condition={self.condition}, true_branch={self.true_branch}, false_branch={self.false_branch})"
    
class ImportNode(AST):
    """Represents an import statement."""
    def __init__(self, module_name):
        self.module_name = module_name

    def __repr__(self):
        return f"ImportNode(module_name={self.module_name})"
    
class PythonNode(AST):
    """Represents a block of Python code."""
    def __init__(self, code: str) -> None:
        self.code = code

    def __repr__(self) -> str:
        return f"PythonNode(code={self.code})"


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
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def error(self, message: str) -> None:
        token = self.current_token
        raise SyntaxError(f"{Fore.RED}{Style.BRIGHT}SyntaxError:{Style.RESET_ALL} "
                          f"{message} at token {token} (pos {token.pos})")

    def eat(self, token_type: str) -> None:
        if self.current_token.type == token_type:
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
        else:
            self.error(f"Expected token {token_type}, got {self.current_token.type}")

    def import_statement(self) -> ImportNode:
        """Parses an import statement: import IDENTIFIER.IDENTIFIER"""
        self.eat("IMPORT")
        if self.current_token.type != "IDENTIFIER":
            self.error("Expected module name after 'import'")
        module_name = self.current_token.value
        self.eat("IDENTIFIER")
        if self.current_token.type == "DOT":
            self.eat("DOT")
            if self.current_token.type != "IDENTIFIER":
                self.error("Expected module name after '.'")
            module_name += f".{self.current_token.value}"
            self.eat("IDENTIFIER")
        return ImportNode(module_name)

    def python_block(self) -> PythonNode:
        """Parses a Python block (now a single PYTHON_BLOCK token)."""
        token = self.current_token
        self.eat("PYTHON_BLOCK")  # Consume the PYTHON_BLOCK token
        return PythonNode(token.value)  # Directly use the token's value
        

    def factor(self) -> AST:
        """Parses numbers, strings, booleans, variables (or function calls), and expressions in parentheses."""
        token = self.current_token

        if token.type == "NUMBER":
            self.eat("NUMBER")
            return Num(token)

        elif token.type == "STRING":
            self.eat("STRING")
            return Str(token)

        elif token.type == "BOOL":
            self.eat("BOOL")
            return Bool(token)

        elif token.type == "IDENTIFIER":
            self.eat("IDENTIFIER")
            if self.current_token.type == "LPAREN":  # Function call
                self.eat("LPAREN")
                args = []
                if self.current_token.type != "RPAREN":
                    args.append(self.expr())
                    while self.current_token.type == "COMMA":
                        self.eat("COMMA")
                        args.append(self.expr())
                self.eat("RPAREN")
                return FunctionCall(token.value, args)
            else:
                return VarNode(token.value)  # Variable reference

        elif token.type == "LPAREN":
            self.eat("LPAREN")
            node = self.expr()
            self.eat("RPAREN")
            return node

        else:
            self.error(f"Expected NUMBER, STRING, BOOL, variable, or '(' Got {token.type}")

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
        """Parses a function definition:
           func IDENTIFIER ( [params] ) { {statement} }
        """
        self.eat("FUNC")
        if self.current_token.type != "IDENTIFIER":
            self.error("Expected function name after 'func'")
        func_name = self.current_token.value
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        parameters = []
        if self.current_token.type != "RPAREN":
            if self.current_token.type != "IDENTIFIER":
                self.error("Expected parameter name")
            parameters.append(self.current_token.value)
            self.eat("IDENTIFIER")
            while self.current_token.type == "COMMA":
                self.eat("COMMA")
                if self.current_token.type != "IDENTIFIER":
                    self.error("Expected parameter name")
                parameters.append(self.current_token.value)
                self.eat("IDENTIFIER")
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

    def statement(self) -> AST:
        """Parses a statement: print, assignment, function def, or return."""
        if self.current_token.type == "PRINT":
            self.eat("PRINT")
            exprs = [self.expr()]  # Parse the first expression
            while self.current_token.type == "COMMA":  # Handle additional expressions separated by commas
                self.eat("COMMA")
                exprs.append(self.expr())
            return PrintNode(exprs)
        
        elif self.current_token.type == "IMPORT":
            return self.import_statement()

        elif self.current_token.type == "FUNC":
            return self.function_def()

        elif self.current_token.type == "RETURN":
            return self.return_statement()

        elif self.current_token.type == "IDENTIFIER":  # Check for function call as statement.
            token = self.current_token
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
                return FunctionCall(token.value, args) # Return function call node directly
            
            elif self.current_token.type == "ASSIGN":
                self.eat("ASSIGN")
                expr_node = self.expr()
                return AssignNode(token.value, expr_node)
            
            else:
                self.error("Unknown statement or missing assignment operator")

        elif self.current_token.type == "WHILE":
            self.eat("WHILE")
            condition = self.comparison()  # Use comparison() instead of expr()
            body = self.block()
            return WhileNode(condition, body)
        
        elif self.current_token.type == "IF":
                self.eat("IF")
                condition = self.comparison()
                self.eat("LBRACE")  # Expect '{'
                true_branch = []
                while self.current_token.type != "RBRACE":
                    true_branch.append(self.statement())
                self.eat("RBRACE")  # Expect '}'

                false_branch = None
                if self.current_token.type == "ELSE":
                    self.eat("ELSE")
                    self.eat("LBRACE")
                    false_branch = []
                    while self.current_token.type != "RBRACE":
                        false_branch.append(self.statement())
                    self.eat("RBRACE")

                return IfNode(condition, true_branch, false_branch)
        
        elif self.current_token.type == "PYTHON":  # Handle embedded Python code
            python_code = self.python_block()
            return PythonNode(python_code)

        else:
            self.error("Unknown statement")

    def parse(self) -> List[AST]:
        """Parse the token list and return the AST for all statements."""
        statements = []
        while self.current_token.type != "EOF":
            stmt = self.statement()
            statements.append(stmt)
        return statements


# -------------------------------
#          Interpreter
# -------------------------------
class ReturnValue(Exception):
    """Custom exception to handle function return values."""
    def __init__(self, value: Any):
        self.value = value


class Interpreter:
    def __init__(self):
        self.variables = {}  # Dictionary to store variable values
        self.functions = {}  # Dictionary to store functions

    def visit(self, node):
        if isinstance(node, PrintNode):
            values = [self.visit(expr) for expr in node.exprs]  # Evaluate all expressions
            print(*values)

        elif isinstance(node, ImportNode):
            if node.module_name == "math":
               factor = """
func factorial(n) {
    if n == 0 {
        return 1
    } else {
        return n * factorial(n - 1)
    }
}"""           
               fibon = """

func fibonacci(n) {
    if n == 0 {
        return 0
    } 
    
    if n == 1 {
        return 1
        
    } else {
        return fibonacci(n - 1) + fibonacci(n - 2)
    }
}


"""
               sin = """

func sine(x) {
    return x - (x * x * x) / 6 + (x * x * x * x * x) / 120
    }

"""
               self.functions['factorial'] = Parser(Lexer(factor).tokenize()).parse()[0]
               self.functions['fibonacci'] = Parser(Lexer(fibon).tokenize()).parse()[0]
               self.functions['sin'] = Parser(Lexer(sin).tokenize()).parse()[0]
                

            if node.module_name == 'lfunc':
                print(self.functions)
            
            if node.module_name == 'lvar':
                print(self.variables)

            if node.module_name == 'runtime':
                print(dir(self))

            if '.' in node.module_name:
                with open(node.module_name, 'r') as f:
                    code = f.read()
                    lexer = Lexer(code)
                    tokens = lexer.tokenize()
                    parser = Parser(tokens)
                    tree = parser.parse()
                    for stmt in tree:
                        if isinstance(stmt, FunctionDef):
                            self.functions[stmt.func_name] = stmt

        elif isinstance(node, AssignNode):
            value = self.visit(node.expr)
            self.variables[node.var_name] = value

        elif isinstance(node, VarNode):
            if node.var_name in self.variables:
                return self.variables[node.var_name]
            else:
                raise NameError(f"Undefined variable '{node.var_name}'")

        elif isinstance(node, BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
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
            return node.value

        elif isinstance(node, Bool):
            return node.value

        elif isinstance(node, FunctionDef):
            self.functions[node.func_name] = node

        elif isinstance(node, FunctionCall):
            if node.func_name == "len":
                if len(node.args) != 1:
                    raise TypeError("len() expects exactly 1 argument")
                arg = self.visit(node.args[0])
                if not isinstance(arg, str):
                    raise TypeError("len() expects a string argument")
                return len(arg)
            
            elif node.func_name == "upstr":
                if len(node.args) != 1:
                    raise TypeError("upper() expects exactly 1 argument")
                arg = self.visit(node.args[0])
                if not isinstance(arg, str):
                    raise TypeError("upper() expects a string argument")
                return arg.upper()
            
            elif node.func_name == 'rstr':
                if len(node.args) != 1:
                    raise TypeError("reverse() expects exactly 1 argument")
                arg = self.visit(node.args[0])
                if not isinstance(arg, str):
                    raise TypeError("reverse() expects a string argument")
                return arg[::-1]
            
            elif node.func_name == 'lowstr':
                if len(node.args) != 1:
                    raise TypeError("lower() expects exactly 1 argument")
                arg = self.visit(node.args[0])
                if not isinstance(arg, str):
                    raise TypeError("lower() expects a string argument")
                return arg.lower()
            
            elif node.func_name == 'str':
                if len(node.args) != 1:
                    raise TypeError("str() expects exactly 1 argument")
                arg = self.visit(node.args[0])
                if isinstance(arg, str):
                    return arg
                elif isinstance(arg, (int, float)):
                    return str(arg)
                else:
                    raise TypeError("str() expects a string, int or float argument")
                
            elif node.func_name == 'type':
                if len(node.args) != 1:
                    raise TypeError("type() expects exactly 1 argument")
                arg = self.visit(node.args[0])
                if isinstance(arg, str):
                    return "str"
                elif isinstance(arg, (int, float)):
                    return "int"
                elif isinstance(arg, bool):
                    return "bool"
                else:
                    return "unknown"
                
            elif node.func_name == 'int':
                if len(node.args) != 1:
                    raise TypeError("int() expects exactly 1 argument")
                arg = self.visit(node.args[0])
                if isinstance(arg, str):
                    return int(arg)
                elif isinstance(arg, (int, float)):
                    return int(arg)
                else:
                    raise TypeError("int() expects a string, int or float argument")
                
            elif node.func_name == 'input':
                if len(node.args) != 1:
                    raise TypeError("input() expects exactly 1 argument")
                arg = self.visit(node.args[0])
                if not isinstance(arg, str):
                    raise TypeError("input() expects a string argument")
                return input(arg)

            if node.func_name not in self.functions:
                raise NameError(f"Undefined function '{node.func_name}'")
            func_def = self.functions[node.func_name]
            args = [self.visit(arg) for arg in node.args]
            if len(args) != len(func_def.parameters):
                raise TypeError("Function argument count mismatch")
            # Save the current variable scope and create a new one
            old_vars = self.variables.copy()
            for param, arg_value in zip(func_def.parameters, args):
                self.variables[param] = arg_value
            try:
                for stmt in func_def.body:
                    self.visit(stmt)
                ret_value = None
            except ReturnValue as rv:
                ret_value = rv.value
            # Restore the previous scope
            self.variables = old_vars
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
            condition_result = self.visit(node.condition)
            if condition_result:
                for stmt in node.true_branch:
                    self.visit(stmt)
            elif node.false_branch:
                for stmt in node.false_branch:
                    self.visit(stmt)

        elif isinstance(node, PythonNode):
            try:
                exec(node.code, globals(), self.variables)
            except Exception as e:
                print(f"Error in embedded Python code: {e}") # Or handle the error as needed
        
    def interpret(self, tree: List[AST]) -> None:
        """Interpret each statement in the AST."""
        for stmt in tree:
            self.visit(stmt)

# C Transpile
# Mapping from our token types (used in BinOp and CompareNode) to C operators.
OPERATOR_MAP = {
    "PLUS": "+",
    "MINUS": "-",
    "MUL": "*",
    "DIV": "/",
    "MOD": "%",
    "AND": "&",
    "XOR": "^",
    "LT": "<",
    "LE": "<=",
    "GT": ">",
    "GE": ">=",
    "EQ": "==",
    "NE": "!="
}

class CodeGenerator:
    def __init__(self):
        self.global_output = []        # Lines of code generated at global scope.
        self.main_output = []          # Lines of code to be placed inside main.
        self.indent_level = 0          # Current indentation level.
        self.declared_vars = set()     # Track declared variables (for each scope).

    def indent(self):
        return "    " * self.indent_level

    def function_return_type(self, node):
        """Determine the return type of a function by checking for ReturnNodes in the body."""
        # A simple check: if any immediate statement is a ReturnNode, we return int.
        for stmt in node.body:
            if isinstance(stmt, ReturnNode):
                return "int"
        return "void"

    def visit(self, node):
        """Recursively generate C code from AST nodes."""
        # Assignment: Declare variable on first assignment.
        if isinstance(node, AssignNode):
            expr_code = self.visit(node.expr)
            # If the expression is a string literal, use char*; otherwise, assume int.
            if isinstance(node.expr, Str):
                var_type = "char*"
            else:
                var_type = "int"
            if node.var_name not in self.declared_vars:
                self.declared_vars.add(node.var_name)
                return f"{self.indent()}{var_type} {node.var_name} = {expr_code};"
            else:
                return f"{self.indent()}{node.var_name} = {expr_code};"

        # Variable reference.
        elif isinstance(node, VarNode):
            return node.var_name

        # Number literal.
        elif isinstance(node, Num):
            return str(node.value)

        # String literal.
        elif isinstance(node, Str):
            return f"\"{node.value}\""

        # Boolean literal.
        elif isinstance(node, Bool):
            return "1" if node.value else "0"

        # Binary operations.
        elif isinstance(node, BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op = OPERATOR_MAP.get(node.op.type, node.op.type)
            return f"({left} {op} {right})"

        # Comparison nodes (if defined separately as CompareNode).
        elif hasattr(node, "left") and hasattr(node, "op") and hasattr(node, "right") and node.__class__.__name__ == "CompareNode":
            left = self.visit(node.left)
            right = self.visit(node.right)
            op = OPERATOR_MAP.get(node.op.type, node.op.type)
            return f"({left} {op} {right})"

        # Print statement.
        elif isinstance(node, PrintNode):
            expr_code = self.visit(node.expr)
            # If printing a string literal, use %s; otherwise, assume integer.
            if isinstance(node.expr, Str):
                return f'{self.indent()}printf("%s\\n", {expr_code});'
            else:
                return f'{self.indent()}printf("%d\\n", {expr_code});'

        # If statement (with optional else).
        elif isinstance(node, IfNode):
            result = f"{self.indent()}if ({self.visit(node.condition)}) {{\n"
            self.indent_level += 1
            for stmt in node.true_branch:
                result += self.visit(stmt) + "\n"
            self.indent_level -= 1
            result += f"{self.indent()}}}"
            if node.false_branch:
                result += " else {\n"
                self.indent_level += 1
                for stmt in node.false_branch:
                    result += self.visit(stmt) + "\n"
                self.indent_level -= 1
                result += f"{self.indent()}"
            return result

        # While loop.
        elif isinstance(node, WhileNode):
            result = f"{self.indent()}while ({self.visit(node.condition)}) {{\n"
            self.indent_level += 1
            for stmt in node.body:
                result += self.visit(stmt) + "\n"
            self.indent_level -= 1
            result += f"{self.indent()}}}"
            return result

        # Function definition.
        elif isinstance(node, FunctionDef):
            ret_type = self.function_return_type(node)
            params = ", ".join(f"int {p}" for p in node.parameters)
            result = f"{ret_type} {node.func_name}({params}) {{\n"
            self.indent_level += 1
            # New local scope for function: clear declared vars.
            old_declared = self.declared_vars.copy()
            self.declared_vars = set()
            for stmt in node.body:
                result += self.visit(stmt) + "\n"
            self.indent_level -= 1
            result += f"{self.indent()}}}"
            self.declared_vars = old_declared
            return result

        # Function call.
        elif isinstance(node, FunctionCall):
            args = ", ".join(self.visit(arg) for arg in node.args)
            return f"{node.func_name}({args})"

        # Return statement.
        elif isinstance(node, ReturnNode):
            return f"{self.indent()}return {self.visit(node.expr)};"

        else:
            raise Exception(f"Unknown node type: {type(node)}")

    def generate(self, ast):
        """
        Generate C code from the AST.
          - Function definitions are output at the global scope.
          - Other statements go inside main().
        """
        global_lines = ["#include <stdio.h>", ""]
        main_lines = ["int main() {"]
        self.indent_level = 1  # For main
        self.declared_vars = set()  # Reset declared variables for main

        # First, output global function definitions.
        for node in ast:
            if isinstance(node, FunctionDef):
                global_lines.append(self.visit(node))
                global_lines.append("")  # Add an empty line

        # Next, output the rest of the statements inside main.
        for node in ast:
            if not isinstance(node, FunctionDef):
                main_lines.append(self.visit(node))
        main_lines.append(f"{self.indent()}return 0;")
        main_lines.append("}")
        return "\n".join(global_lines + main_lines)




# -------------------------------
#              Main
# -------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python parser.py <filename>")
        return

    try:
        with open(sys.argv[1], "r", encoding="utf-8") as file:
            source = file.read()

        # Tokenization
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Parsing
        parser = Parser(tokens)
        tree = parser.parse()

        # codegen = CodeGenerator()
        # c_code = codegen.generate(tree)

        # Interpretation
        interpreter = Interpreter()
        interpreter.interpret(tree)

        # Presentation AST
        def pretty_ast(node, indent=0):
            prefix = " " * (indent * 3)
            if isinstance(node, AssignNode):
                return f"{prefix}Assign(\n{prefix}    var={repr(node.var_name)},\n{prefix}    value={pretty_ast(node.expr, indent + 1)}\n{prefix})"
            elif isinstance(node, PrintNode):
                exprs_str = ",\n".join(pretty_ast(expr, indent + 1) for expr in node.exprs)
                return f"{prefix}Print(\n{prefix}    exprs=[\n{exprs_str}\n{prefix}    ]\n{prefix})"
            elif isinstance(node, BinOp):
                return f"{prefix}BinOp(\n{prefix}    left={pretty_ast(node.left, indent + 1)},\n{prefix}    op='{node.op.type}',\n{prefix}    right={pretty_ast(node.right, indent + 1)}\n{prefix})"
            elif isinstance(node, Num):
                return f"{prefix}Num(value={node.value})"
            elif isinstance(node, Str):
                return f"{prefix}Str(value={repr(node.value)})"
            elif isinstance(node, Bool):
                return f"{prefix}Bool(value={node.value})"
            elif isinstance(node, VarNode):
                return f"{prefix}Var(name={repr(node.var_name)})"
            elif isinstance(node, FunctionDef):
                body_str = "\n".join(pretty_ast(stmt, indent + 1) for stmt in node.body)
                return f"{prefix}FunctionDef(name={node.func_name}, params={node.parameters}, body=[\n{body_str}\n{prefix}])"
            elif isinstance(node, FunctionCall):
                args_str = ", ".join(pretty_ast(arg, indent + 1) for arg in node.args)
                return f"{prefix}FunctionCall(name={node.func_name}, args=[{args_str}])"
            elif isinstance(node, ReturnNode):
                return f"{prefix}Return({pretty_ast(node.expr, indent + 1)})"
            elif isinstance(node, CompareNode):  # Added CompareNode support
                    return f"{prefix}Compare(\n{prefix}    left={pretty_ast(node.left, indent + 1)},\n{prefix}    op='{node.op.type}',\n{prefix}    right={pretty_ast(node.right, indent + 1)}\n{prefix})"
            elif isinstance(node, WhileNode):  # Added WhileNode support
                    body_str = "\n".join(pretty_ast(stmt, indent + 1) for stmt in node.body)
                    return f"{prefix}While(\n{prefix}    condition={pretty_ast(node.condition, indent + 1)},\n{prefix}    body=[\n{body_str}\n{prefix}])"
            elif isinstance(node, IfNode):  
                true_branch_str = "\n".join(pretty_ast(stmt, indent + 1) for stmt in node.true_branch)
                false_branch_str = "\n".join(pretty_ast(stmt, indent + 1) for stmt in node.false_branch) if node.false_branch else "None"
                return f"{prefix}If(\n{prefix}    condition={pretty_ast(node.condition, indent + 1)},\n{prefix}    true_branch=[\n{true_branch_str}\n{prefix}    ],\n{prefix}    false_branch=[\n{false_branch_str}\n{prefix}    ]\n{prefix})"
            else:
                return f"{prefix}{repr(node)}"
        
            
        iosnb_filename = f"{sys.argv[1].split('.')[0]}.iosnb"
        mode = "w" if os.path.exists(iosnb_filename) else "x"
        with open(iosnb_filename, mode, encoding="utf-8") as file:
            file.write(f"{tokens}\n\n")
            for stmt in tree:
                file.write(pretty_ast(stmt) + "\n")
            # file.write('\n'+c_code)   
            # with open(f"{sys.argv[1].split('.')[0]}.c", "w" if os.path.exists(iosnb_filename) else "x", encoding="utf-8") as c:
            #     c.write(c_code)
            # os.system(f"fgcc {sys.argv[1].split('.')[0]}.c")
            # os.system(f"{sys.argv[1].split('.')[0]}")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()