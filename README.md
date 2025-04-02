# iso
The ISO Programming Language

![Description of the image](https://github.com/CoataocCreate/iso/blob/main/iso.png?raw=true)

# ISO Language Documentation

## Overview
The ISO language is a simple interpreted language that includes basic syntax for variables, functions, control flow, and embedded Python code. The language is designed to be easy to understand and extend.

## Tokens
ISO language uses the following tokens:

- **NUMBER**: Represents numerical literals.
- **STRING**: Represents string literals enclosed in single quotes.
- **BOOL**: Represents boolean values (`True` or `False`).
- **IDENTIFIER**: Represents variable names and function names.
- **PRINT**: Keyword for printing expressions.
- **FUNC**: Keyword for defining functions.
- **RETURN**: Keyword for returning values from functions.
- **WHILE**: Keyword for while loops.
- **IF**: Keyword for if statements.
- **ELSE**: Keyword for else statements.
- **IMPORT**: Keyword for import statements.
- **PLUS (+)**: Represents addition.
- **MINUS (-)**: Represents subtraction.
- **MUL (*)**: Represents multiplication.
- **DIV (/)**: Represents division.
- **MOD (%)**: Represents modulo operation.
- **AND (&)**: Represents bitwise AND.
- **XOR (^)**: Represents bitwise XOR.
- **LT (<)**: Represents less than.
- **LE (<=)**: Represents less than or equal to.
- **GT (>)**: Represents greater than.
- **GE (>=)**: Represents greater than or equal to.
- **EQ (==)**: Represents equality.
- **NE (!=)**: Represents inequality.
- **ASSIGN (=)**: Represents assignment.
- **LPAREN (()**: Represents left parenthesis.
- **RPAREN ())**: Represents right parenthesis.
- **LBRACE ({)**: Represents left brace.
- **RBRACE (})**: Represents right brace.
- **COMMA (,)**: Represents a comma.
- **DOT (.)**: Represents a dot.

## Lexer
The lexer converts an input string into a stream of tokens. It handles whitespace, numbers, strings, identifiers, operators, and reserved keywords.

## Parser
The parser implements a recursive descent parser to convert the token stream into an Abstract Syntax Tree (AST). The grammar for the parser includes:

- **statement**: `PRINT expr | IDENTIFIER ASSIGN expr | func IDENTIFIER LPAREN [params] RPAREN LBRACE {statement} RBRACE | RETURN expr`
- **expr**: `term ((PLUS | MINUS) term)*`
- **term**: `factor ((MUL | DIV | MOD | AND | XOR) factor)*`
- **factor**: `NUMBER | STRING | BOOL | IDENTIFIER [function_call] | LPAREN expr RPAREN`

## AST Nodes
The AST nodes represent different constructs in the ISO language:

- **PrintNode**: Represents a print statement.
- **AssignNode**: Represents variable assignment.
- **Num**: Represents numerical literals.
- **Str**: Represents string literals.
- **Bool**: Represents boolean literals.
- **BinOp**: Represents binary operations.
- **VarNode**: Represents variable references.
- **FunctionDef**: Represents function definitions.
- **FunctionCall**: Represents function calls.
- **CompareNode**: Represents comparison operations.
- **WhileNode**: Represents while loops.
- **ReturnNode**: Represents return statements.
- **IfNode**: Represents if-else statements.
- **ImportNode**: Represents import statements.
- **PythonNode**: Represents embedded Python code blocks.

## Interpreter
The interpreter executes the AST by visiting each node and performing the corresponding actions. It supports variable assignment, function definitions, function calls, print statements, control flow, and import statements.

## Code Generator
The code generator translates the AST into C code. It handles global function definitions, variable declarations, binary operations, comparison operations, print statements, if-else statements, while loops, function definitions, function calls, and return statements.

## Main
The main function reads the source code from a file, tokenizes it, parses it into an AST, interprets the AST, and generates the corresponding C code.

## Example Usage
To use the ISO language, create a file with ISO code and run the interpreter:

```shell
python parser.py <filename>
```

This will interpret the code and optionally generate C code if enabled in the script.

For more detailed examples and usage, refer to the source code in the `iso.py` file.

---

For more information, visit the [ISO language repository](https://github.com/CoataocCreate/iso).
