#include <stdio.h>
#include "tokens.h"

extern FILE *yyin;
extern int yylex();
extern char *yytext;

int main() {
    // Test with simple input
    yyin = fopen("test_input.txt", "w");
    fprintf(yyin, "let x = 5\n");
    fprintf(yyin, "def add a b = a + b");
    fclose(yyin);
    
    yyin = fopen("test_input.txt", "r");
    
    int token;
    while ((token = yylex()) != 0) {
        printf("Token: %d, Text: %s\n", token, yytext);
    }
    
    fclose(yyin);
    return 0;
}
