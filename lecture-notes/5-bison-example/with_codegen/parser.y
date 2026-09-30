%{
#include <stdio.h>
#include <math.h>
#include "ast.h"
#include "codegen.h"

int yylex(void);
void yyerror(char const *);
ASTNode *ast_root = NULL;
%}

%union {
    ASTNode *node;
    double dval;
}

%token <dval> NUM
%type <node> exp

%left '-' '+'
%left '*' '/'
%precedence NEG
%right '^'

%%

input:
    %empty
    | input line
;

line:
    '\n'
    | exp '\n' { 
        ast_root = $1;
        
        printf("=== Expression Evaluation ===\n");
        
        // Method 1: Direct AST evaluation
        printf("1. AST Evaluation: %g\n", ast_evaluate(ast_root));
        
        // Method 2: Code generation + execution
        printf("2. Code Generation:\n");
        Bytecode *bytecode = codegen_generate(ast_root);
        codegen_disassemble(bytecode);
        codegen_execute(bytecode);
        codegen_free(bytecode);
        
        // Show AST structure
        printf("3. AST Structure:\n");
        ast_print(ast_root, 0);
        
        // Cleanup
        ast_free(ast_root);
        ast_root = NULL;
        printf("\n");
    }
;

exp:
    NUM { $$ = ast_create_number($1); }
    | exp '+' exp { $$ = ast_create_binary(AST_ADD, $1, $3); }
    | exp '-' exp { $$ = ast_create_binary(AST_SUB, $1, $3); }
    | exp '*' exp { $$ = ast_create_binary(AST_MUL, $1, $3); }
    | exp '/' exp { $$ = ast_create_binary(AST_DIV, $1, $3); }
    | '-' exp %prec NEG { $$ = ast_create_unary(AST_NEG, $2); }
    | exp '^' exp { $$ = ast_create_binary(AST_POW, $1, $3); }
    | '(' exp ')' { $$ = $2; }
;

%%

void yyerror(char const *s) {
    fprintf(stderr, "Error: %s\n", s);
}

int main(void) {
    printf("Calculator with AST and Code Generation\n");
    printf("Enter expressions or Ctrl-D to exit\n\n");
    yyparse();
    return 0;
}