{
  open Printf
  
  type token = 
    | LET 
    | DEF 
    | IDENTIFIER of string
    | NUMBER of int
    | PLUS 
    | EQUALS 
    | EOF
    | ERROR of char

  let string_of_token = function
    | LET -> "LET"
    | DEF -> "DEF" 
    | IDENTIFIER s -> sprintf "IDENTIFIER(%s)" s
    | NUMBER n -> sprintf "NUMBER(%d)" n
    | PLUS -> "PLUS"
    | EQUALS -> "EQUALS"
    | EOF -> "EOF"
    | ERROR c -> sprintf "ERROR('%c')" c
}

rule token = parse
| [' ' '\t']     { token lexbuf }
| '\n'           { Lexing.new_line lexbuf; token lexbuf }
| "let"          { printf "KEYWORD 'let' at line %d\n" (Lexing.lexeme_start lexbuf |> fst); LET }
| "def"          { printf "KEYWORD 'def' at line %d\n" (Lexing.lexeme_start lexbuf |> fst); DEF }
| ['a'-'z' 'A'-'Z']['a'-'z' 'A'-'Z' '0'-'9' '_']* as id
    { printf "IDENTIFIER '%s' at line %d\n" id (Lexing.lexeme_start lexbuf |> fst); 
      IDENTIFIER id }
| ['0'-'9']+ as num
    { printf "NUMBER %s at line %d\n" num (Lexing.lexeme_start lexbuf |> fst);
      NUMBER (int_of_string num) }
| '+'            { printf "OPERATOR '+' at line %d\n" (Lexing.lexeme_start lexbuf |> fst); PLUS }
| '='            { printf "OPERATOR '=' at line %d\n" (Lexing.lexeme_start lexbuf |> fst); EQUALS }
| eof            { EOF }
| _ as c         { printf "ERROR: unexpected '%c' at line %d\n" c (Lexing.lexeme_start lexbuf |> fst); ERROR c }