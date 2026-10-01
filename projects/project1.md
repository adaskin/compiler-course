# BIL 463 — Assignment 1: Lexical Analysis

**Fall 2026 · Ammar Daşkın**
**Due:** Monday, October 20, 2026 (before class)
**Submit:** Google Classroom (Part A PDF + Part B zip file with REPORT.pdf)

---

## Overview

| Part | What | Weight | Mode |
|------|------|--------|------|
| **A** | Theory: RE → NFA → DFA (handwritten) | 30% | Individual |
| **B** | Design & implement a lexer for your own language + BPE comparison | 60% | Group (2–3) |
| **Bonus** | RE→NFA→DFA visualization tool | +10% | Group (2–3) |

---

## AI Policy

AI tools (ChatGPT, Copilot, DeepSeek, etc.) are **permitted and encouraged**. But:

- **Part A must be handwritten.** Use AI to study, but submit your own work on paper.
- **Document AI usage in your report.** Which tools, what prompts, what it got wrong.
- **You must be able to explain your code in class.** I will pick groups in Week 5 and ask questions live.
- **If you used AI and did not mention it, that is an academic integrity issue.**

---

## Part A: Theory (Individual) — 30%

**Handwritten, scanned/photographed as one PDF.** Name and student ID on every page.

---

### A1. Regular Expressions (15 pts)

**A1.1** (10 pts) — Write a regex for each language. Give 3 accepted and 3 rejected strings.

(a) Over {a, b}: strings with an even number of `a`'s

(b) Over {0, 1}: binary strings starting with `1`, ending with `0`

(c) C identifiers: `[a-zA-Z_][a-zA-Z0-9_]*`

(d) C integers: optional sign, no leading zeros (except `0`)

(e) Floats: `3.14`, `.5`, `2.`, `1e10`, `2.5E-3` valid; `.`, `e5`, `1.2.3` invalid

**A1.2** (5 pts) — Describe L(s) in English. Give 3 accepted and 3 rejected strings.

(a) `[a-zA-Z_][a-zA-Z0-9_]*`

(b) `0x[0-9a-fA-F]+`

(c) `"([^"\\]|\\.)*"`

---

### A2. Finite Automata Construction (15 pts)

Draw a finite automaton (NFA or DFA) accepting each language:

**(a)** (5 pts) {w ∈ {a, b}* | w starts with 'a' and contains 'baba' as a substring}

**(b)** (5 pts) {w ∈ {0, 1}* | w contains '111' and does not contain '00'}

**(c)** (5 pts) {w ∈ {a, b, c}* | #a mod 2 = #b mod 3}

---

### A3. Thompson's Construction (10 pts)

Draw the NFA, showing construction steps:

**(a)** (5 pts) `a(b|c)*d`

**(b)** (5 pts) `[0-9]+(\.[0-9]+)?`

---

### A4. Subset Construction + Minimization (15 pts)

**A4.1** (8 pts) — Convert this NFA to a DFA. Show the complete table.

```
NFA over Σ = {a, b}
States: {0, 1, 2}    Start: 0    Accept: {2}
δ(0, a) = {0, 1}     δ(0, b) = {0}
δ(1, b) = {2}
δ(2, a) = {2}        δ(2, b) = {2}
```

Show: each DFA state as a named subset, complete transition table, accepting states, final DFA diagram.

**A4.2** (7 pts) — RE Equivalence via Minimized DFAs

Check whether each pair is equivalent by constructing minimized DFAs:

**(i)** (4 pts) `(0 | 1)*` and `(0* | 10*)*`

**(ii)** (3 pts) `(ba)+(a*b* | a*)` and `(ba)*ba+(b* | ε)`

---

### A5. Short Answer (5 pts)

3–5 sentences each.

**(a)** (2 pts) Why does maximal munch cause `ifcondition` to be one IDENTIFIER instead of KEYWORD + IDENTIFIER? What would minimal munch break?

**(b)** (3 pts) In BPE tokenization, what plays the role of "rule priority" from Flex? How does merge order resolve ambiguity?

---

## Part B: Design & Implement Your Own Language (Group) — 60%

### Step 1: Define Your Language

Design a small programming language. You have creative freedom, but it **must** include:

| Required Feature | Example |
|-----------------|---------|
| At least 6 keywords | `let`, `if`, `while`, `return`, ... |
| Identifiers | Clear rules (start with letter? underscore?) |
| Integer and float literals | `42`, `3.14` |
| At least 8 operators | Including multi-character (`==`, `<=`, `&&`) |
| Punctuation | Parens, braces, semicolons, etc. |
| Single-line comments | `// ...` or `# ...` |
| Multi-line comments | `/* ... */` or similar |
| At least one control structure | `if`, `while`, `for` |
| At least one loop | `while`, `for`, `loop` |

**You decide:**
- Language name and purpose (general purpose? DSL for math? agent tasks?)
- Significant whitespace or not?
- Keywords reserved or not?
- Unusual tokens? (`@decorator`, `->` arrow, `|>` pipe)

---

### Step 2: Write Language Specification

Before coding, write a **1-page spec** including:
- Language name and one-sentence description
- Complete token table (category → regex → example)
- 3 example programs (5-10 lines each)
- 2 design decisions and WHY you made them

---

### Step 3: Implement the Lexer

Choose one path:

| Path | Tools |
|------|-------|
| A | C + Flex |
| B | Python + PLY |
| C | OCaml + OCamllex |

Your lexer must:
- Tokenize all defined tokens correctly
- Handle **maximal munch** (`==` is one token, not two `=`)
- Handle **keyword priority** (`if` → KEYWORD, not IDENTIFIER)
- Track line numbers
- Handle errors gracefully (report line, skip, continue)
- Handle multi-character operators
- Skip comments

---

### Step 4: Write Test Suite

Since your language is unique, **you write your own tests:**
- At least **3 test files**
- At least **15 distinct test cases** total
- Must include: valid programs, edge cases, error cases
- Show expected output for each test

---

### Step 5: BPE Comparison

Tokenize your test files with `tiktoken` and compare:

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")

with open("tests/test1.txt") as f:
    text = f.read()

bpe_tokens = enc.encode(text)
print(f"BPE: {len(bpe_tokens)} tokens")
print([enc.decode([t]) for t in bpe_tokens[:20]])
```

---

### Deliverables

Submit a **zip file** containing:

```
project1/
├── README.md           # Group members, path, build instructions, AI tools used
├── SPEC.pdf            # Language specification (1 page)
├── src/                # Your lexer code
├── tests/              # Your test files (at least 3)
├── output/             # Expected output for each test
└── REPORT.pdf          # See below
```

---

### REPORT.pdf (2 pages, 3 sections)

**Section 1: Language Design** (½ page)
- Why did you choose these keywords and operators?
- How do you handle multi-character operators (e.g., `=` vs `==` vs `<=`)?
- Walk through what happens with input `ifx` in your language.

**Section 2: Testing & Bugs** (½ page)
- List 3 specific bugs you encountered. For each: what failed, why, how you fixed it.
- What edge cases did you test? Any surprises?

**Section 3: AI Usage + BPE Comparison** (1 page)

*AI usage:*
- Which tools? Include 2–3 prompts verbatim.
- What did AI get wrong? What did you fix?
- If no AI used, write: "No AI tools were used."

*BPE comparison:*
1. Your lexer vs BPE: how many tokens each? Why the difference?
2. How does BPE handle `42`? Your lexer? Why?
3. How does BPE handle error characters (e.g., `@`)?
4. Which tokenization is better for a **parser**? For a **neural network**?
5. Tokenize `"Derleyici belirteçleri ayrıştırır"` with BPE. How many tokens? Why more than English?
   


---

## Bonus: Visualization Tool (+10%)

**Optional.** Build a CLI tool that takes a regex and outputs:
1. NFA diagram (Thompson's)
2. DFA diagram (subset construction)
3. Minimized DFA (Hopcroft's)
4. String trace (accepted/rejected)

Support: `|`, `*`, `+`, `?`, concatenation, `(...)`, `[a-z]`

**Recommended:** Python + graphviz.

```bash
$ python re2dfa.py "a(b|c)*d" --trace "abbd"
NFA: 8 states → DFA: 4 states → Minimized: 3 states
Trace: A --a--> B --b--> B --b--> B --d--> C [ACCEPTED]
```
 Put code in `bonus/` directory. Include sample diagrams.

**Optional Cross-Domain Regex Investigation**  
> Choose a discipline outside CS. Formulate a small research question answerable by extracting lexical patterns from a public corpus. Use AI to brainstorm regexes and edge cases. Manually verify 20+ examples. Compute precision and recall. Visualize your results. Explain the limits of regex for your question in REPORT.pdf.

---

## Grading

### Part A (30%)
| Item | Pts |
|------|-----|
| A1: Regex | 15 |
| A2: FA construction | 15 |
| A3: Thompson's | 10 |
| A4: Subset + minimization | 15 |
| A5: Short answer | 5 |

### Part B (60%)
| Item | Pts |
|------|-----|
| Language spec (clear, complete, justified) | 15 |
| Correct tokenization | 15 |
| Maximal munch + keyword priority + error handling | 10 |
| Test suite (≥3 files, ≥15 cases) | 10 |
| Report quality (all 3 sections) | 10 |
| BPE comparison (in Section 3) | 10 |

### Bonus (+10%)
| Item | Pts |
|------|-----|
| Correct algorithms | 6 |
| Clear diagrams | 2 |
| String tracing | 2 |
| Or/and cross-Domain Regex Investigation | 10 | 

---


## FAQ

**Can I use Flex/PLY/OCamllex?**
Yes, that's the point.

**What if my lexer misses edge cases?**
Partial credit. Get common cases right.

**Do I need GPU for BPE?**
No. `pip install tiktoken` runs on CPU.

**Can I use Rust/Java?**
With approval. Rust+`logos` or Java+JFlex are fine.

**My group has 2 people. OK?**
Yes.

**How detailed should Thompson's steps be?**
Show base cases, combination, final NFA.

**For A2(c), how many states?**
Think: `#a mod 2` (2 values) × `#b mod 3` (3 values) = 6 states.

---

## Resources

- Flex: https://westes.github.io/flex/manual/
- PLY: https://ply.readthedocs.io/en/latest/
- OCamllex: https://ocaml.org/manual/lexyacc.html
- tiktoken: https://github.com/openai/tiktoken
- Graphviz: https://graphviz.org/
- BPE paper: https://arxiv.org/abs/1508.07909

---

*Start with Part A. The theory makes Part B make sense.*