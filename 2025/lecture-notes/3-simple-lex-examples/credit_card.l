/*http://web.eecs.utk.edu/~bvanderz/teaching/cs461Sp11/notes/flex/*/
%option noyywrap
%{
/* * * * * * * * * * * *
 * * * DEFINITIONS * * *
 * * * * * * * * * * * */
%}

%{
// recognize whether or not a credit card number is valid
int line_num = 1;
%}

digit [0-9]
group {digit}{4}
%%

%{
/* * * * * * * * * 
 * * * RULES * * *
 * * * * * * * * */
%}
   /* The carat (^) says that a credit card number must start at the
      beginning of a line and the $ says that the credit card number
      must end the line. */
^{group}([ -]?{group}){3}$  { printf(" credit card number: %s\n", yytext); }

   /* The .* accumulates all the characters on any line that does not
      match a valid credit card number */
.* { printf("%d: error: %s \n", line_num, yytext); }
\n { line_num++; }
%%

/* * * * * * * * * * * 
 * * * USER CODE * * *
 * * * * * * * * * * *
 */
int main(int argc, char *argv[]) {
  yylex();
}