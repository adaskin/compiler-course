%{
#include <stdio.h>
#include <stdlib.h>
#include "type.h"  // Add this line to define Type
#include "parser.tab.h"

int line_number = 1;
void yyerror(const char* s);

%}

%option noyywrap

DIGIT    [0-9]
LETTER   [a-zA-Z_]
ID       {LETTER}({LETTER}|{DIGIT})*
INTEGER  {DIGIT}+
FLOAT    {DIGIT}+\.{DIGIT}*
WS       [ \t\r]

%%

{WS}     /* skip whitespace */
\n       { line_number++; }

"int"    { return TOK_INT; }
"float"  { return TOK_FLOAT; }
"bool"   { return TOK_BOOL; }
"true"   { return TOK_TRUE; }
"false"  { return TOK_FALSE; }

"+"      { return TOK_PLUS; }
"-"      { return TOK_MINUS; }
"*"      { return TOK_MULTIPLY; }
"/"      { return TOK_DIVIDE; }
"="      { return TOK_ASSIGN; }
";"      { return TOK_SEMICOLON; }
"("      { return TOK_LPAREN; }
")"      { return TOK_RPAREN; }

{INTEGER} { 
    yylval.int_val = atoi(yytext);
    return TOK_INTEGER; 
}

{FLOAT}   { 
    yylval.float_val = atof(yytext);
    return TOK_FLOAT; 
}

{ID}      { 
    yylval.string_val = strdup(yytext);
    return TOK_ID; 
}

<<EOF>>  { return TOK_EOF; }

.         { 
    printf("Error: Invalid character '%c' at line %d\n", yytext[0], line_number);
    return TOK_ERROR;
}

%%