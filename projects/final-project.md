# BIL 463 — Final Project: A Functional Language That Humans and LLMs Can Both Read

**Fall 2026 · Ammar Daşkın**
**Submit:** GitHub repository + Google Classroom (link, tags, and `REPORT.pdf`)
**Due dates:** Phase 1 **[TBD]** · Phase 2 **[TBD]** · Phase 3 **[TBD]** · Report + presentation **[TBD]**

---

## Overview

You will design and build a complete compiler front-to-back for **your own small functional programming language (FPL)**. The compiler reads your source, builds an AST, type-checks it, and emits **LLVM IR** that `clang`/`llc` turns into a running program.

One twist: the language must be **easy for humans to read and easy for LLMs to work with**. In 2026 a language is used by people *and* by code-generating models, so you will design for both, and then measure how well an LLM does with your language.

The project is one continuous codebase delivered in **three phases**. Each phase is graded when it is submitted, and each builds on the previous one.

| Phase | What | Points |
|-------|------|--------|
| **1** | Language design + lexer + parser + AST + syntax-error recovery | 20 |
| **2** | Type checker + semantic errors + multi-error recovery | 25 |
| **3** | LLVM IR code generation + end-to-end tests | 30 |
| **Report** | `REPORT.pdf` (including the LLM-friendliness evaluation) | 15 |
| **Presentation** | 10-minute demo + Q&A in the last class | 10 |
| | **Total** | **100** |

**Teams:** groups of **2–3** (same rule as Project 1). A team of 1 is allowed with instructor approval. Every team member must be able to explain any part of the code in the presentation. Each report includes a one-paragraph contribution statement, and individual grades may be adjusted from it and from the git history.

**Reuse:** you may start from your Project 1 lexer, your Project 2/3 code, and the lecture examples (`lecture-notes/5-bison-example`, `lecture-notes/8-semantic-analysis-type-checking`). You may change your language freely; it does not have to match anything you built earlier.

---

## AI Policy

AI tools are **permitted and encouraged**, as in Project 1. That includes using an LLM to help write the compiler. But:

- Document AI usage in `REPORT.pdf`: which tools, 2–3 prompts verbatim, what it got wrong, what you fixed.
- AI-generated tests count only if you have checked that the *expected output is actually correct*.
- You must be able to explain every line live. I will ask.
- Undisclosed AI use is an academic integrity issue.

---

## 1. Fixed vs. Free Design Choices

### Fixed (mandatory for everyone)

| # | Requirement |
|---|-------------|
| F1 | The language must be **functional**: expression-oriented (no statements; `if` and `let` are expressions), **immutable bindings** (no assignment), **recursion** instead of loops, and **functions as values** (a function can be passed as an argument and called). |
| F2 | The compiler builds an explicit **AST** and can dump it (`--emit-ast`). |
| F3 | The compiler **type-checks** statically. Programs that fail the check must not reach code generation. |
| F4 | The compiler emits **LLVM IR** (textual `.ll` is fine and recommended) that passes `opt -verify` and runs correctly after `clang file.ll`. |
| F5 | Tests must cover **errors, correctness, and recovery** (Section 6). |
| F6 | **Minimum language core:** `Int`, `Bool`, arithmetic and comparison, `let`, `if`, top-level function definitions with recursion, function calls, and a built-in `print_int(Int) -> Int` (prints the value and a newline, returns the value; this avoids needing a `Unit` type or IO monad). A `main() -> Int` function is the entry point; its result is the process exit code. |
| F7 | One **feature track** from Section 2, declared by the end of Phase 1. |
| F8 | The common **command-line interface** and **diagnostic format** below, so one script can test every team's compiler. |
| F9 | Public GitHub repo, `README.md` with build instructions, a short `SPEC.md`, and a `tests/` directory. |

**Command-line interface (fixed):**

```bash
./fplc prog.fpl --emit-tokens      # optional in Phase 1, token list
./fplc prog.fpl --emit-ast         # Phase 1+, AST as an S-expression (see Section 3)
./fplc prog.fpl --emit-typed-ast   # Phase 2+, AST with a type on every expression
./fplc prog.fpl --emit-ll -o prog.ll   # Phase 3
```

(`fplc` may be a shell wrapper around `python3 ...` or an OCaml executable; name your language and tool as you like, but keep this interface.)

**Diagnostics (fixed format, written to stderr):**

```
<file>:<line>:<col>: error[<code>]: <one-line message>
```

Exit code `0` on success, `1` if any error was reported, and never a crash, hang, or segfault on any input, however malformed.

### Free (your design)

Syntax style and keywords, the names of everything, the type syntax, which extra types you add, operators and precedence, how functions are declared, whether `let` is `let ... in` or something else, how you parse (recursive descent, Flex/Bison, PLY, OCaml `ocamllex`/Menhir), the structure of your AST, the error-recovery strategy, the IR-generation strategy, and any additional features or optimizations.

**Implementation paths** are the same three as the rest of the course (C + Flex/Bison, Python + PLY, OCaml + ocamllex/Menhir). Emit **textual LLVM IR** and call `clang`/`llc`; you do not need LLVM bindings in any path. See `lecture-notes/10-code-generation.md`.

---

## 2. Feature Tracks (pick at least one)

Declare your track at the end of Phase 1 (in `SPEC.md`). It affects Phases 2 and 3 (type rules and codegen), so choose early. Tracks are listed roughly from least to most work. A bigger track earns no extra points by itself; it just gives you more room to score the 6 "feature track" points in Phase 3 and makes the project more interesting.

| Track | You add | Hard part |
|-------|---------|-----------|
| **A. Tuples / records + lists** | pair/tuple or record types, a list type with `cons`/`head`/`tail`/`is_empty` (or equivalent), a heap allocation via `malloc` | Type rules for compound types; layout in LLVM |
| **B. Algebraic data types + pattern matching** | `type Option = None \| Some(Int)`, `match` expressions | Constructor typing, tags in memory, lowering `match` to branches (bonus: exhaustiveness check) |
| **C. Closures** | `fn(x: Int) -> Int = x + n` anonymous functions that capture variables | Environment capture, closure conversion/lambda lifting, closure representation in LLVM |
| **D. Polymorphism / inference** | `let` without annotations, generic functions (Hindley–Milner or a simpler scheme) | Unification; instantiating generics in codegen (monomorphization) |

Heap memory may simply be allocated with `malloc` and never freed. (A garbage collector or reference counting is a bonus; see `lecture-notes/lecture17-auto-mem-management.pdf`.)

---

## 3. Scope Calibration: A Minimal Example

This is **one possible** language, shown only so you can judge the expected amount of work. Do not copy it; design your own.

### Example program (`example.fpl`)

```
// Higher-order helper: apply f twice
fn apply_twice(f: (Int) -> Int, x: Int) -> Int = f(f(x))

fn square(x: Int) -> Int = x * x

fn fact(n: Int) -> Int =
  if n <= 1 then 1 else n * fact(n - 1)

fn main() -> Int =
  let a = fact(5) in
  let b = apply_twice(square, 3) in
  if print_int(a + b) == 201 then 0 else 1
```

Output: `201` (120 + 81), exit code `0`.

### Example grammar (the whole language fits on one screen)

```ebnf
program   = { fn_decl } ;
fn_decl   = "fn" IDENT "(" [ param { "," param } ] ")" "->" type "=" expr ;
param     = IDENT ":" type ;
type      = "Int" | "Bool" | "(" [ type { "," type } ] ")" "->" type ;
expr      = "let" IDENT [ ":" type ] "=" expr "in" expr
          | "if" expr "then" expr "else" expr
          | or_expr ;
or_expr   = and_expr { "||" and_expr } ;
and_expr  = cmp_expr { "&&" cmp_expr } ;
cmp_expr  = add_expr [ ("==" | "!=" | "<" | "<=" | ">" | ">=") add_expr ] ;
add_expr  = mul_expr { ("+" | "-") mul_expr } ;
mul_expr  = unary { ("*" | "/" | "%") unary } ;
unary     = ("-" | "!") unary | call ;
call      = primary { "(" [ expr { "," expr } ] ")" } ;
primary   = INT | "true" | "false" | IDENT | "(" expr ")" ;
```

### `--emit-ast` for `fact`

```
(fn fact ((n Int)) Int
  (if (<= (var n) (int 1))
      (int 1)
      (* (var n) (call (var fact) (- (var n) (int 1))))))
```

### A type error and the message it should produce

```
bad.fpl:1:26: error[E201]: type mismatch in function body
  expected: Bool   (declared return type of 'bad')
  found:    Int
   1 | fn bad(x: Int) -> Bool = x + 1
     |                          ^^^^^
  help: change the return type to Int, or make the body a Bool expression
```

(Only the first line is the fixed format. Everything after it is what makes a message *good*; see Section 7.)

### LLVM IR for `fact` (what Phase 3 produces)

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

Functions passed as arguments are plain `ptr` values pointing to LLVM functions (`call i32 %f(i32 %x)`), so requirement F1 (functions as values) needs **no closures**. Closures are Track C.

### How big is this?

The example above (about 12 grammar rules, 2 types plus function types, 8 AST node kinds) is the **baseline** that a team of 2–3 should reach comfortably. As a rough guide, not a requirement: expect on the order of a thousand lines of OCaml or Python, or a few thousand in C. Your feature track and your diagnostics are where the rest of your time goes.

---

## 4. Phases and Deliverables

Every phase is submitted as a **git tag** (`phase1`, `phase2`, `phase3`) plus a Classroom note with the repo link. Tests are cumulative: Phase 3 must still pass Phase 1 and 2 tests.

### Phase 1: Language Design, Parser, AST (20 pts)

**Build**
- `SPEC.md` (max 3 pages): informal description, the grammar (EBNF), at least **8 example programs** of increasing difficulty, and your declared feature track.
- Lexer (you may reuse Project 1) and parser producing an AST.
- `--emit-ast` printing the AST in an S-expression (or other regular, indented) format.
- **Syntax-error recovery:** after a syntax error, resynchronize (e.g., skip to the next top-level `fn`, or to a closing token) and keep parsing, so one run reports **several independent syntax errors**.

**Deliverables:** `src/`, `SPEC.md`, `README.md`, `tests/parse/` (see Section 6), git tag `phase1`.

| Rubric | Pts |
|--------|-----|
| `SPEC.md`: clear, complete grammar, examples, design justification | 5 |
| Parser accepts all valid programs; correct precedence and associativity | 6 |
| AST design and `--emit-ast` output | 4 |
| Syntax errors: located, readable, with recovery | 3 |
| Tests (Section 6 minimums) | 2 |

### Phase 2: Type Checker (25 pts)

**Build**
- Symbol table and scoping (`let`, parameters, top-level functions; mutual recursion between top-level functions should work, or you document why not).
- A type checker over the AST for your whole language, including function types and your feature track.
- `--emit-typed-ast`: AST annotated with types.
- **Multi-error reporting without cascades:** after a type error in an expression, give it a special `error` type that is compatible with everything, so a single mistake does not produce ten follow-on errors. Report all independent errors in one run.
- At least these checks: undefined variable/function, wrong argument count, wrong argument type, `if` condition not `Bool`, branch types differ, return type mismatch, operator type mismatch, duplicate definition, missing `main` or wrong `main` signature.

**Deliverables:** updated `SPEC.md` with a **type-rules section** (typing rules or a precise table), `tests/type/`, git tag `phase2`.

| Rubric | Pts |
|--------|-----|
| Type rules documented (inference-rule or table form) | 4 |
| Checker accepts all well-typed programs, rejects ill-typed ones (including your track) | 8 |
| Diagnostic quality (Section 7: location, expected/found, hint) | 5 |
| Recovery: several errors per run, no cascades, no crash | 4 |
| Tests (Section 6 minimums) | 4 |

### Phase 3: LLVM Code Generation (30 pts)

**Build**
- Code generator from the **typed** AST to textual LLVM IR (`--emit-ll`), plus a tiny runtime/driver (`print_int` via `printf` is enough).
- A `Makefile` or script target such as `make run FILE=prog.fpl` that compiles and runs the program with `clang`.
- Correct lowering of `let`, `if` (branches + `phi`), calls, recursion, function values, and your feature track.
- A test runner (`make test` or `./run_tests.sh`) that runs *all* tests from all phases.

**Deliverables:** `src/`, `runtime/`, `tests/run/`, `make test`, git tag `phase3`.

| Rubric | Pts |
|--------|-----|
| Valid IR: every emitted `.ll` passes `opt -verify`; no codegen on ill-typed input | 6 |
| Correct results on the core language (arithmetic, `let`, `if`, recursion, function values) | 10 |
| Feature track implemented and correct | 6 |
| End-to-end tests and test runner (Section 6 minimums) | 4 |
| Robustness: no crashes on bad input, sensible exit codes, deep recursion behaves | 4 |

**Optional bonus (up to +10 total, added to the project score):** tail-call elimination (`musttail` or loop lowering) (+3) · constant folding or another simple optimization pass with before/after IR (+2) · code formatter/pretty-printer where `format(format(x)) == format(x)` and the AST is preserved (+3) · `--diagnostics=json` machine-readable errors (+2) · garbage collection or reference counting (+3) · an editor syntax-highlighting file (+1). Cap +10.

---

## 5. Suggested Milestones

A comfortable plan is **7 weeks**, with the last week for polish, report, and presentations (if the calendar is shorter, shrink Phase 3, not the tests). Replace the week numbers with real dates once announced.

| Week | Milestone |
|------|-----------|
| **1** | Form teams, pick path and tools, design the language, write the first draft of `SPEC.md` and 8 example programs. 10-minute check-in with me. |
| **2** | Parser + AST + `--emit-ast` + syntax-error recovery + parser tests. **Phase 1 due.** Declare your track. |
| **3** | Symbol table, core type rules, `--emit-typed-ast`. |
| **4** | Type errors, recovery without cascades, your track's typing, type tests. **Phase 2 due.** |
| **5** | Codegen for the core (`int`/`bool`/`let`/`if`/calls/recursion) and `print_int`; the first program runs. |
| **6** | Function values, feature track, full test runner, `opt -verify` on all tests. **Phase 3 due.** |
| **7** | LLM-friendliness experiment (Section 7), report, slides, dry run. Presentations in class. |

**Advice for a small team:** get a *vertical slice* working early (a program like `fn main() -> Int = 42` through every stage), then widen. Do not leave the LLVM step to the end; try writing the IR by hand for your examples in Week 2 so you know what you are aiming at.

---

## 6. Test Requirements

Tests are graded and are **the evidence** that your compiler does what you say. Put them under `tests/` and make them runnable with one command.

### Layout

```
tests/
├── parse/    prog.fpl  prog.expected      # expected AST dump or expected diagnostics
├── type/     prog.fpl  prog.expected      # expected typed AST or diagnostics
└── run/      prog.fpl  prog.expected      # expected stdout; prog.exit holds the exit code if nonzero
```

### The three required categories

| Category | What it shows | Example |
|----------|---------------|---------|
| **Correctness** | Valid programs compile, and run to the right answer | `fact(5)` prints `120`; `apply_twice(square, 3)` is `81` |
| **Errors** | Invalid programs are *rejected* with the right diagnostic (right line, right code, no crash) | Missing `then`; `x + true`; call with 2 args to a 1-argument function |
| **Recovery** | One file with *several* mistakes produces *several* diagnostics, in the right order, with no bogus follow-on errors, and the compiler terminates normally | A file with 3 unrelated syntax errors reports exactly 3; a file with 2 independent type errors reports exactly 2 |

### Minimums

| Phase | Correctness | Errors | Recovery |
|-------|-------------|--------|----------|
| 1 (parse) | 8 programs (AST matches) | 6 syntax-error programs | 3 multi-error files |
| 2 (type) | 8 well-typed programs (typed AST matches) | 10 type-error programs (one per check in Phase 2's list) | 3 multi-error files, incl. one that *would* cascade without an `error` type |
| 3 (run) | 12 programs (output + exit code), covering recursion, function values, and your track | 3 programs that must *not* reach codegen | 2 files where earlier errors do not hide later ones |

Also required:
- **Every emitted `.ll` in `tests/run/` passes `opt -verify`** (the test runner checks it).
- At least **one "program of your own"** of 40+ lines (e.g., primes, fibonacci with an accumulator, a small list or tree algorithm for your track) that shows the language is actually usable.
- Test names are descriptive (`type_if_branch_mismatch.fpl`, not `test7.fpl`).
- **Robustness check:** truncating each valid test program at 5 random points (script it) must never crash the compiler; every run ends with a diagnostic or success.
- The README documents how to run the tests and shows the pass count.

---

## 7. LLM-Friendliness Criteria

This is what makes the project about *2026*. LLMs read your `SPEC.md`, write programs in your language, and read your compiler's error messages to repair them. Your design is judged against the criteria below; they are scored inside the Phase 1 and 2 rubrics (spec, diagnostics) and in the report (evaluation).

| Criterion | What it means | Evidence to give |
|-----------|---------------|------------------|
| **Regular syntax** | Few constructs, one canonical way to write each, every construct explicitly delimited (`in`, `end`, parentheses). Avoid whitespace-sensitive or context-dependent parsing. The grammar fits on one page. | The grammar in `SPEC.md`; a sentence on any place where it is *not* regular and why |
| **Explicit types** | Function signatures fully annotated (parameter and return types). No implicit coercions (e.g., `Int` → `Bool`), no overloading. Local inference is fine, but a reader never has to guess a signature. | The type rules; examples |
| **Locality** | You can understand a function from the function itself plus the signatures it calls. No hidden global state, no implicit imports, no action at a distance. | A short argument in the report |
| **Good error messages** | Each diagnostic has: stable **error code**, **file:line:col**, a message that names the problem, **expected vs. found**, the offending source line with a caret, and a **one-line fix hint** where one exists. Deterministic, and no cascades. | Diagnostics in `tests/`; the `bad.fpl` style shown in Section 3 |
| **Inspectable stages** | An LLM or a human can look at tokens, the AST, the typed AST, and the IR, each in a regular text format. | `--emit-*` flags |
| **Prompt-sized spec** | `SPEC.md` is short and example-driven enough to paste into a prompt. | Page count; number of examples |

### The LLM experiment (required, goes in the report)

Measure how well an LLM works with your language, using only your `SPEC.md` as the language description:

1. Write **10 small programming tasks** (e.g., "sum of digits", "gcd", "count down and print", plus 2 using your track).
2. Give an LLM the `SPEC.md` and each task, with no other help. Save the prompts and outputs verbatim in `llm-eval/`.
3. Compile each result. Record: **compiles?**, **runs correctly?**, and for failures, the error category (syntax, type, logic).
4. **Repair loop:** feed the compiler's diagnostic back to the LLM for one retry. How many failures are fixed by the message alone?
5. Change *one thing* in your spec, grammar, or messages based on what you learned, and re-measure.
6. As in Project 1, report the **BPE token count** (`tiktoken`) for your example programs, and compare it to an equivalent short Python or OCaml program.

Your score does **not** depend on a high LLM success rate. It depends on whether you measured honestly and drew a sensible, specific conclusion about your design choices.

---

## 8. Final Report and Presentation

### `REPORT.pdf` (4 pages max, 15 pts)

| Section | Content | Pts |
|---------|---------|-----|
| 1. Language design | Goals; 3 design decisions and the alternatives you rejected; how the design serves humans *and* LLMs | 4 |
| 2. Architecture & testing | Pipeline diagram; how recovery works in the parser and the checker; 3 real bugs (what failed, why, fix); test counts | 3 |
| 3. LLM-friendliness evaluation | The experiment from Section 7: table of results, the one change you made, before/after, conclusion | 5 |
| 4. AI usage & contributions | Tools used, 2–3 verbatim prompts, what AI got wrong; per-member contribution paragraph | 3 |

### Presentation (10 minutes + questions, 10 pts)

In the last class. Cover: your language in one slide, a **live demo** (a valid program through to its output, a program with errors and the diagnostics, and the AST/typed AST/IR flags), the hardest bug, the LLM result, and what you would do differently.

| Rubric | Pts |
|--------|-----|
| Clear explanation of design and pipeline | 4 |
| Working live demo (have a recorded fallback) | 3 |
| Answers to questions show every member understands the code | 3 |

---

## 9. Grading Summary and Policies

| Component | Points |
|-----------|--------|
| Phase 1 | 20 |
| Phase 2 | 25 |
| Phase 3 | 30 |
| Report | 15 |
| Presentation | 10 |
| Bonus | up to +10 (total capped at 100 + 10) |

- **No late submissions** for phases (same as Project 1). A phase that is late loses the phase score, but later phases can still be submitted.
- If a phase is incomplete, submit what works; partial credit is given. List what is missing in the README.
- Plagiarism policy as in the syllabus. Discussing ideas is fine; code must be your team's own (AI usage disclosed).

---

## FAQ

**Does my language have to look like the example?** No. Keep the *requirements*; change the *syntax* and anything else you like. A Lisp-like S-expression syntax is a perfectly good choice; it is very regular and makes parsing easy, but then your effort moves to types and diagnostics.

**Can I use Hindley–Milner inference?** Yes (Track D), but function signatures should stay annotated or at least be printable by the compiler, so a reader sees them.

**Do I need strings or floats?** No. `Int` and `Bool` are enough. Add them if your track needs them.

**Can I use the LLVM C API / OCaml `llvm` bindings?** Yes. Textual IR is simpler and works in all paths.

**What about memory management?** `malloc` without `free` is acceptable. Mention it in the report.

**My language has no loops. Is that OK?** That is the point: recursion replaces loops. Tail-call elimination is a bonus.

**Can I fix Phase 1 or 2 bugs after the due date?** Yes, in the final submission; they will be reflected in the test runner and the report, but the phase score stays as graded.

---

## Resources

- Course examples: `lecture-notes/5-bison-example/`, `lecture-notes/8-semantic-analysis-type-checking/`, `lecture-notes/10-code-generation.md`
- LLVM Language Reference: https://llvm.org/docs/LangRef.html
- LLVM Kaleidoscope tutorial (codegen ideas): https://llvm.org/docs/tutorial/
- Crafting Interpreters (free online): https://craftinginterpreters.com
- Types and Programming Languages (type systems), Pierce
- `opt -verify`, `llc`, `clang file.ll -o prog`
- tiktoken: https://github.com/openai/tiktoken

---

*Get a program like `fn main() -> Int = 42` through every stage in the first week. Then grow the language.*
