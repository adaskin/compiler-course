%{
#include <stdio.h>
#include "tokens.h"

int line_num = 1;
%}

DIGIT    [0-9]
LETTER   [a-zA-Z_]
WS       [ \t]

%%
\n       { line_num++; }
{WS}+    /* skip whitespace */

"let"    { printf("LINE %d: KEYWORD 'let'\n", line_num); return LET; }
"def"    { printf("LINE %d: KEYWORD 'def'\n", line_num); return DEF; }
"+"      { printf("LINE %d: OPERATOR '+'\n", line_num); return PLUS; }
"="      { printf("LINE %d: OPERATOR '='\n", line_num); return EQUALS; }
{LETTER}({LETTER}|{DIGIT})* { 
    printf("LINE %d: IDENTIFIER '%s'\n", line_num, yytext); 
    return IDENTIFIER; 
}

{DIGIT}+ { 
    printf("LINE %d: NUMBER '%s'\n", line_num, yytext); 
    return NUMBER; 
}

.        { printf("LINE %d: ERROR: unexpected '%s'\n", line_num, yytext); }
%%

int yywrap() { return 1; }
