%{
#include "ast.h"           // Include AST definitions first

#include "parser.tab.h"
#include <stdlib.h>
void yyerror(char const *);
%}

%%

[0-9]+(\.[0-9]*)?   { 
    yylval.dval = atof(yytext);  /* Set the dval field of the union */
    return NUM; 
}
"+"                 { return '+'; }
"-"                 { return '-'; }
"*"                 { return '*'; }
"/"                 { return '/'; }
"^"                 { return '^'; }
"("                 { return '('; }
")"                 { return ')'; }
\n                  { return '\n'; }
[ \t]               ; /* ignore whitespace */
.                   { yyerror("Invalid character"); }

%%

int yywrap(void) {
    return 1;
}