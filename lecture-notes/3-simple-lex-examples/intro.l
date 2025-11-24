/*https://westes.github.io/flex/manual/Simple-Examples.html#Simple-Examples*/
%{
    /* C Preamble Code */
%}
    /*Definitions*/
%%
username    printf( "%s", getlogin() );

%%
 /* run this on terminal 
 $ flex intro.l
 $ gcc lex.yy.c -ll
 $ ./a.out 
 */
