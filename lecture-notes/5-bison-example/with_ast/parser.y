%{
#include <stdio.h>
#include <math.h>
#include "ast.h"

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
        printf("Result: %g\n", ast_evaluate(ast_root));
        printf("AST:\n");
        ast_print(ast_root, 0);
        ast_free(ast_root);
        ast_root = NULL;
    }
;

exp:
    NUM { $$ = ast_create_number($1); }  /* $1 is double from lexer */
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
    printf("AST Calculator\n");
    yyparse();
    return 0;
}