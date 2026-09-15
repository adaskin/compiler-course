let () =
  let test_input = "let x = 5\ndef add a b = a + b" in
  let lexbuf = Lexing.from_string test_input in
  
  print_endline "=== OCAML LEXER DEMO ===";
  
  let rec print_tokens lexbuf =
    match Minilang.token lexbuf with
    | Minilang.EOF -> 
        print_endline "End of input"
    | token ->
        print_endline (Minilang.string_of_token token);
        print_tokens lexbuf
  in
  
  print_tokens lexbuf