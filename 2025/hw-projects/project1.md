project 1

You can work in groups of 2 or 3 or individually.
Describe the specifications of a programming language 
You can either define a completely new PL
At least, there should be keywords, identifiers, comments, conditions, loops, and operations
You will use this for the 2nd and 3rd projects
Or use a minimal version of The Go Programming Language or another language 
At least, there should be common keywords, identifiers, comments, conditions, loops, and operations


Using lexer (flex or another tool), write a scanner for the language you described in 1: it should be able to read in source on the input and should output token types (as in the last example in the lecture). Test it out by applying it to different source files you have created.




Hw2---giving in classroom

1. (taken from engineering a compiler): Construct  FAs for accepting each of the following languages:
 {w ∈ {a, b}∗ | w starts with ‘a’ and contains ‘baba’ as a substring}
 {w ∈ {0, 1}∗ | w contains ‘111’ as a substring and does not contain ‘00’ as a substring}
 {w ∈ {a, b, c}∗ | in w the number of ‘a’s modulo 2 is equal to the number of ‘b’s modulo 3}
2. One way of proving that two REs are equivalent is to construct their minimized DFAs and then compare them. If they differ only by state names, then the res are equivalent. Use this technique to check the following pairs of REs and state whether or not they are equivalent.
 (0 | 1)∗ and (0∗ | 10∗ )∗
(ba)+ (a∗ b∗ | a∗ ) and (ba)∗ ba+ (b∗ | )
