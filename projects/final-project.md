# BIL 463 — Final Project: A Functional Language That Humans and LLMs Can Both Read

**Fall 2026 · Ammar Daşkın**
**Groups:** 2–3 students · **One project, three phases, graded per phase**
**Checkpoints:** Phase 1 **[TBD]** · Phase 2 **[TBD]** · Phase 3 **[TBD]** · Report + demo **[TBD]**
**Submit:** GitHub repo (one git tag per phase) + Google Classroom note with the link

---

## 1. The Goal

Design a small **functional programming language** and build a compiler that:

1. Lexes and parses it into an **AST**
2. **Type checks** it and reports errors with locations
3. Generates **LLVM IR** that actually runs
4. Is validated by tests that show **correctness, errors, and recovery**

You choose the language's name, syntax, and personality. You do **not** choose whether to type check or generate real code; those are the point of the project.

**The twist:** the language must be easy for **humans to read** and easy for **LLMs to work with** (Section 6). You will design for both, then measure how well an LLM does with your language.

It is one continuous codebase. Phases build on each other, and each phase is graded when you reach it. You may reuse your Project 1 lexer and the lecture examples (`lecture-notes/5-bison-example`, `lecture-notes/8-semantic-analysis-type-checking`).

---

## 2. Language Requirements

### Must support

| Feature | Requirement |
|---------|-------------|
| Types | `int`, `bool`, and function types `T -> T` |
| Literals | Integers and `true`, `false` |
| Arithmetic | `+`, `-`, `*`, `/` on integers (and unary `-`) |
| Comparison | `<`, `<=`, `>`, `>=`, `==`, `!=` on integers; `==`, `!=` on bools |
| Boolean operators | `&&`, `\|\|` (short-circuit) |
| Conditionals | `if e1 then e2 else e3`, an **expression**; `else` is mandatory; both branches have the same type |
| Local bindings | `let x = e1 in e2`, lexical scope, shadowing allowed |
| Named functions | Top-level `let f (x: T) (y: T): T = e`, recursive, fully annotated |
| Anonymous functions | `fun (x: int) -> e`, one annotated parameter, **no capture** of local variables (see below) |
| Function values | A function can be passed as an argument and called: `apply double 5` |
| Built-ins | `print_int : int -> int` and `print_bool : bool -> int` (print the value and a newline, return 0) |

### Program shape

A program is a sequence of top-level definitions followed by **one final expression**. Top-level function definitions may refer to each other (including mutual recursion). Top-level non-function definitions (`let answer = ...`) are evaluated in order, before the final expression. The final expression is the body of LLVM `@main`; its value is discarded and `main` returns 0. Programs are tested by their **stdout**.

### Required type rules

| Rule | Condition |
|------|-----------|
| `if e1 then e2 else e3 : T` | `e1 : bool`, `e2 : T`, `e3 : T` |
| `let x = e1 in e2 : T` | `e1 : S`, and `e2 : T` with `x : S` added to the environment |
| `fun (x: S) -> e : S -> T` | `e : T` with `x : S` added to the environment |
| `f a : T` | `f : S -> T` and `a : S` |
| Arithmetic, comparison, `&&`, `\|\|`, `-` | operands and results as in the table above |
| Unbound variable | **error** |
| `int` vs `bool` | **no implicit conversion** |
| `==` / `!=` on functions | **error** |

Function parameters and return types are **annotated** (no type inference required). A local `let x = e1 in ...` needs no annotation; its type is the type of `e1`. Annotating anonymous-function parameters is the price of not requiring inference: it is a deliberate, defensible design decision, so explain it in your report.

### Fixed semantics (so expected outputs are comparable across teams)

- `int` is 32-bit signed and wraps on overflow.
- Evaluation is **strict, left-to-right, call-by-value** (this matters because `print_int` has side effects).
- A call supplies **all** of a function's arguments at once. **No partial application.**
- Function **return types are `int` or `bool`** (a function may take functions as arguments but not return one).
- A lambda may not use variables of an enclosing scope. Top-level names are not "captured". A capturing lambda must be rejected with a clear diagnostic (unless you take the closure bonus).
- Integer division or remainder by zero stops the program with a message on stderr and a **nonzero exit code**, never undefined behavior.
- No mutable variables, no assignment, no loops: iteration is recursion. No I/O other than the two print built-ins.
- Any lexical, syntax, or type error means a nonzero exit and **no code generation**. Error recovery exists to report more errors; it must **never turn an invalid program into an accepted one**.

### You may add, but finish it

Lists, tuples, strings, pattern matching, polymorphism. Every feature you add must be parsed, type checked, and code generated, or explicitly rejected with a diagnostic. A half-implemented feature is worse than an absent one.

---

## 3. Scope Calibration: A Minimal Example

This is **one possible** language, shown so you can judge the expected amount of work. The syntax is yours to design. You may keep it, change it, or choose something very different (an S-expression syntax is a fine choice).

```
let double (x: int): int = x * 2

let apply (f: int -> int) (x: int): int = f x

let fact (n: int): int =
  if n <= 1 then 1 else n * fact (n - 1)

let answer =
  let x = 5 in
  let y = apply double x in
  if y > fact 3 then y else fact 3

print_int answer
```

It prints `10` (`double 5 = 10`, `fact 3 = 6`).

### Example grammar (it fits on one screen)

```ebnf
program   = { def } expr ;
def       = "let" IDENT { "(" IDENT ":" type ")" } [ ":" type ] "=" expr ;
type      = atype [ "->" type ] ;
atype     = "int" | "bool" | "(" type ")" ;
expr      = "let" IDENT [ ":" type ] "=" expr "in" expr
          | "if" expr "then" expr "else" expr
          | "fun" "(" IDENT ":" type ")" "->" expr
          | or_expr ;
or_expr   = and_expr { "||" and_expr } ;
and_expr  = cmp_expr { "&&" cmp_expr } ;
cmp_expr  = add_expr [ ( "<" | "<=" | ">" | ">=" | "==" | "!=" ) add_expr ] ;
add_expr  = mul_expr { ( "+" | "-" ) mul_expr } ;
mul_expr  = unary { ( "*" | "/" ) unary } ;
unary     = "-" unary | app ;
app       = atom { atom } ;
atom      = INT | "true" | "false" | IDENT | "(" expr ")" ;
```

Function definitions need all annotations (parameters and result); a plain value definition may omit the type. Notes that your parser will have to settle: application is by juxtaposition (`f x`), so `f -1` is a subtraction, and a top-level `let` without `in` is a definition while one with `in` begins the final expression.

### `--emit-ast` for `fact`

```
(def fact ((n int)) int
  (if (<= (var n) (int 1))
      (int 1)
      (* (var n) (app (var fact) (- (var n) (int 1))))))
```

### A type error and the message it should produce

```
bad.fpl:1:26: error[E201]: type mismatch in function body
  expected: bool   (declared result type of 'bad')
  found:    int
   1 | let bad (x: int): bool = x + 1
     |                          ^^^^^
  help: change the result type to int, or make the body a bool expression
```

Only the first line is the fixed format. Everything after it is what makes a message *good* (Section 6).

### LLVM IR for `fact`

```llvm
define i32 @fact(i32 %n) {
entry:
  %c = icmp sle i32 %n, 1
  br i1 %c, label %then, label %else
then:
  br label %join
else:
  %n1 = sub i32 %n, 1
  %r = call i32 @fact(i32 %n1)
  %m = mul i32 %n, %r
  br label %join
join:
  %v = phi i32 [ 1, %then ], [ %m, %else ]
  ret i32 %v
}
```

Immutable `let` bindings become plain SSA values; no stack slots are needed. Functions passed as arguments are `ptr` values pointing to LLVM functions (`call i32 %f(i32 %x)`), so no closures are required.

### How big is this?

The example above (about 10 grammar rules, 2 base types plus function types, a handful of AST node kinds) is the **baseline** that a team of 2–3 should reach comfortably. As a rough guide, not a requirement: on the order of a thousand lines of OCaml or Python, a few thousand in C. The tests and diagnostics are where the rest of your time goes.

---

## 4. Implementation Paths, CLI, and Diagnostics

| Path | Frontend | Code generation |
|------|----------|-----------------|
| A | C + Flex / Bison | Emit `.ll` text (or LLVM C API) |
| B | Python + PLY | Emit `.ll` text (or `llvmlite`) |
| C | OCaml + ocamllex / Menhir | Emit `.ll` text (or the OCaml LLVM bindings) |

**Emitting readable `.ll` text yourself is encouraged.** You will see the IR you produce. See `lecture-notes/10-code-generation.md`. Build programs with `clang prog.ll runtime.c -o prog` (a minimal C runtime provides the print built-ins). Use an LLVM version recent enough for opaque `ptr` types (LLVM 15+) and say which one you tested with in your README.

You must write the AST, the type checker, and the lowering yourself. Lexer/parser generators and LLVM itself are fine to use. No prebuilt type-checker or parser-combinator library that does the work for you. Your grammar must work with your chosen parser: LALR(1)-compatible (or conflicts resolved explicitly) for Bison/PLY/Menhir, LL(1)-friendly (or backtracking documented) for recursive descent. Submit the grammar with a precedence and associativity table.

**Fixed command-line interface** (so one script can test every team's compiler; the executable may be a wrapper script):

```bash
./fplc prog.fpl --emit-ast         # AST as an S-expression (Phase 1+)
./fplc prog.fpl --emit-typed-ast   # AST with a type on every expression (Phase 2+)
./fplc prog.fpl --emit-ll -o prog.ll   # LLVM IR (Phase 3)
```

**Fixed diagnostic format** (stderr, one per error): `<file>:<line>:<col>: error[<code>]: <one-line message>`. Exit code `0` on success, `1` if any error was reported. The compiler never crashes, hangs, or segfaults on any input.

---

## 5. Phases, Tests, and Timeline

Tag each phase in git (`phase1`, `phase2`, `phase3`). Tests are cumulative: Phase 3 must still pass the Phase 1 and 2 tests.

| Phase | Build |
|-------|-------|
| **1. Spec, parser, AST** | `SPEC.md` (max 3 pages: grammar, tokens, precedence table, type rules, restrictions, 8 example programs); lexer and parser; **AST (not a raw parse tree)**; `--emit-ast`; syntax errors with line/column; **panic-mode recovery** (resynchronize on a token such as `in`, `then`, `else`, `let`, `)` or EOF) so one run reports several independent syntax errors |
| **2. Type checker** | Symbol table with lexical scoping and shadowing; all type rules from Section 2; recursive and mutually recursive functions checked against their declared signatures; capture detection for lambdas; `--emit-typed-ast`; diagnostics with **location, expected type, found type**; **recovery**: report all independent errors in one run, continue with the other definitions after an error, and use an `error` type that is compatible with everything so one mistake does not cascade |
| **3. LLVM codegen** | Textual IR from the typed AST (`--emit-ll`) that passes `opt -verify`; arithmetic, `icmp`, `if` as `br` + `phi`, short-circuit `&&`/`\|\|`, `let` as SSA values, functions and calls, recursion, function values; division-by-zero check; a tiny C runtime; `run_tests.sh` that builds and runs **everything** |

### Test requirements

You write your own tests, since you design your own language. Layout:

```
tests/
├── parse/    prog.fpl  prog.expected    # AST dump or diagnostics
├── type/     prog.fpl  prog.expected    # typed AST or diagnostics
└── run/      prog.fpl  prog.expected    # stdout (prog.exit if exit code is nonzero)
```

| Category | Minimum | What it must show |
|----------|---------|-------------------|
| **Correctness** | 10 programs | Valid programs with the right stdout: precedence, shadowing, recursion, function values, `if` in expression position, short-circuit |
| **Errors** | 5 syntax + 8 type | Each rejected with the right line, right code, and no crash (undefined variable, wrong argument count/type, non-bool condition, branch mismatch, result mismatch, duplicate definition, capturing lambda, ...) |
| **Recovery** | 3 syntax + 3 type files | One file with several mistakes yields several diagnostics, in order, with no bogus follow-on errors; at least one type test would cascade without an `error` type |
| **Runtime** | 1 | Division by zero is diagnosed with a nonzero exit |

Also required:
- **One command** (`make test` or `./run_tests.sh`) builds the compiler, runs all tests, and prints a PASS/FAIL summary.
- **Every emitted `.ll` passes `opt -verify`** (the runner checks it).
- **Do not validate your compiler against itself.** Expected outputs are hand-computed, or produced by a small reference interpreter over your AST (about 50 lines; strongly recommended) and compared with the LLVM program's output.
- At least **one program of your own** of 40+ lines (primes, gcd, fibonacci with an accumulator, ...) that shows the language is usable.
- Descriptive test names (`type_if_branch_mismatch.fpl`, not `test7.fpl`).
- **Robustness:** truncating each valid test program at 5 random points (script it) never crashes the compiler.

### Suggested timeline (about 7 weeks)

| Week | Milestone |
|------|-----------|
| 1 | Form teams, choose path, design the language, draft `SPEC.md`; get a trivial program (`print_int 42`) through every stage in a vertical slice; 10-minute check-in with me |
| 2 | Parser + AST + syntax-error recovery + tests. **Phase 1 due.** |
| 3–4 | Type checker, diagnostics, recovery, tests. **Phase 2 due.** |
| 5–6 | Code generation, function values, full test runner. **Phase 3 due.** |
| 7 | LLM experiment, report, slides, dry run. **Demos in class.** |

Replace the weeks with real dates once announced. Try writing the IR by hand for your examples in week 2, so you know what you are aiming at.

---

## 6. LLM-Friendliness

LLMs read your `SPEC.md`, write programs in your language, and read your compiler's errors to repair them. Your design is judged on these properties:

| Property | What it means |
|----------|---------------|
| **Regular syntax** | Few constructs, one canonical way to write each, everything explicitly delimited (`in`, `then`/`else`, parentheses), no whitespace sensitivity. The grammar fits on one page. |
| **Explicit types** | Fully annotated function signatures; no implicit coercions, no overloading. |
| **Locality** | A function can be understood from itself plus the signatures it calls. No hidden global state. |
| **Good error messages** | Stable error **code**, `file:line:col`, a message naming the problem, **expected vs. found**, the offending line with a caret, and a one-line fix hint where one exists. Deterministic. No cascades. |
| **Inspectable stages** | Tokens, AST, typed AST, and IR are each available as regular text (`--emit-*`). |
| **Prompt-sized spec** | `SPEC.md` is short and example-driven enough to paste into a prompt. |

### LLM experiment (required, goes in the report)

1. Write **10 small tasks** ("gcd", "sum of digits", ...; 2 should use function values).
2. Give an LLM only your `SPEC.md` and one task at a time. Save prompts and outputs verbatim in `llm-eval/`.
3. Compile each. Record: compiles? correct? failure category (syntax, type, logic).
4. **Repair loop:** feed the compiler's diagnostic back once. How many failures does the message alone fix?
5. Change one thing in your spec, grammar, or messages based on what you saw, and re-measure.
6. As in Project 1, compare the BPE token count (`tiktoken`) of your examples with equivalent short Python or OCaml programs.

Your grade does **not** depend on a high success rate, but on honest measurement and a specific conclusion about your design.

---

## 7. Deliverables

```
final-project/
├── README.md        group, path, LLVM version, build/run instructions, test counts
├── SPEC.md          grammar, tokens, precedence, type rules, restrictions, examples
├── src/             lexer, parser, AST, type checker, code generator
├── runtime.c        print built-ins
├── tests/           parse/  type/  run/
├── llm-eval/        prompts, outputs, results
├── examples/        .ll output for 2-3 demo programs
├── run_tests.sh     one-command build + test + summary
└── REPORT.pdf       4 pages max
```

**REPORT.pdf (4 pages):** (1) language design: goals, three design decisions with the alternatives you rejected, how it serves humans and LLMs; (2) architecture and testing: pipeline, how recovery works, three real bugs, test counts, one test that would catch a wrong code-generation rule; (3) the LLM experiment; (4) AI usage (tools, 2–3 prompts verbatim, one concrete AI mistake and how you fixed it) and a contribution paragraph per member. If no AI was used: "No AI tools were used."

**Presentation (10 minutes + questions, last class):** the language in one slide; a live demo (a valid program to its output, a program with errors, and the AST/IR flags); the hardest bug; the LLM result. Each member explains one component. I may introduce a small mutation in your code generator (for example, `add` to `sub`); your test suite should catch it.

**AI policy:** AI tools are permitted and encouraged, including for writing the compiler. Document the usage in the report. AI-generated tests count only if you checked that the expected output is right. You must be able to explain every line live. Undisclosed AI use is an academic integrity issue.

---

## 8. Bonus (up to +10, pick any)

- **Closure conversion:** support capturing lambdas with closure structs and a small runtime.
- **Hindley–Milner inference:** drop annotations, support polymorphic `let`.
- **Lists or algebraic data types** with pattern matching, fully type checked and compiled.
- **Tail-call elimination** (`musttail` or loop lowering).
- **Machine-readable diagnostics** (`--diagnostics=json`).
- Or propose your own extension for approval.

---

## 9. Grading

One project score out of 100, graded per phase. **No late submissions**, as in Project 1. If a phase is incomplete, submit what works and list what is missing in the README; partial credit is given. Bugs from earlier phases can be fixed for the final submission (and will show in the tests and report), but the phase score stays as graded.

| Component | Points | What is graded |
|-----------|--------|----------------|
| **Phase 1** | 20 | `SPEC.md` is clear and complete; parser accepts every valid program with correct precedence and associativity; AST is a real AST; syntax errors are located, readable, and recovered; parser tests |
| **Phase 2** | 25 | Type rules documented; checker accepts well-typed and rejects ill-typed programs; scoping and shadowing; diagnostics have location, expected and found; several errors per run without cascades; type tests |
| **Phase 3** | 30 | Valid IR (`opt -verify`) that runs correctly: arithmetic, `let`, `if`, recursion, function values; division by zero; no crashes on bad input; end-to-end tests, recovery tests, and the test runner |
| **Report** | 15 | Design reasoning; architecture and testing evidence; LLM experiment and conclusion; AI usage and contributions |
| **Presentation** | 10 | Clear explanation, working live demo, every member can answer questions about the code (and the mutation check) |
| **Bonus** | up to +10 | Section 8 |

Individual grades may be adjusted from the contribution paragraph, the git history, and the presentation.

---

## FAQ

**Must we implement everything from scratch?** You write the AST, type checker, and lowering. Flex, Bison, PLY, Menhir, and LLVM are tools you are expected to use.

**Can we skip code generation and just interpret?** No. An interpreter is a useful testing oracle, but LLVM output is required.

**Do we need strings or floats?** No. `int` and `bool` are enough.

**What if we do not finish a feature?** Remove it from the language, or reject it with a diagnostic, and document that it is unsupported.

**Can we use LLVM bindings (`llvmlite`, OCaml `llvm`, the C API)?** Yes, but textual `.ll` is simpler and works in every path.

**What about memory management?** Not needed for the baseline: there is no heap. If you add lists, `malloc` without `free` is fine.

**Do we need a GPU?** No.

---

## Resources

- Course examples: `lecture-notes/5-bison-example/`, `lecture-notes/8-semantic-analysis-type-checking/`, `lecture-notes/10-code-generation.md`
- LLVM Language Reference: https://llvm.org/docs/LangRef.html
- LLVM Kaleidoscope tutorial: https://llvm.org/docs/tutorial/
- Crafting Interpreters: https://craftinginterpreters.com
- `opt -verify`, `llc`, `clang prog.ll runtime.c -o prog`
- tiktoken: https://github.com/openai/tiktoken

---

*Get a trivial program through every stage in the first week. Then grow the language.*
