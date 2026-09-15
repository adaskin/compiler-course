---
marp: true
theme: default
paginate: true
size: 16:10
style: |
  section {
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 24px;
  }
  section.lead {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
  }
  section.lead h1 {
    font-size: 2.5em;
    color: #1a5490;
  }
  h1 { color: #1a5490; }
  h2 { color: #2c5f8d; font-size: 1.4em; }
  h3 { color: #3a7ca5; }
  code {
    background: #f4f4f4;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
  }
  pre code {
    font-size: 0.7em;
    line-height: 1.4;
  }
  table {
    font-size: 0.85em;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .small { font-size: 0.8em; }
  .highlight {
    background: #fff3cd;
    padding: 10px 15px;
    border-left: 4px solid #ffc107;
    margin: 10px 0;
  }
  .ai-callout {
    background: #e7f3ff;
    padding: 10px 15px;
    border-left: 4px solid #1a5490;
    margin: 10px 0;
  }
  .warning {
    background: #f8d7da;
    padding: 10px 15px;
    border-left: 4px solid #dc3545;
    margin: 10px 0;
  }
---

<!-- _class: lead -->

# BIL 463: Compiler Design
## From Foundations to AI Systems

**Fall 2026** · Ammar Daşkın  

Tuesdays 9:30 · 

---

## Welcome 👋

- **Small cohort** (5-10 students) → seminar + lab style
- **Modern twist**: compilers × AI infrastructure
- **Build a real compiler** as your semester project
- **Three implementation paths**: C/Flex/Bison, Python/PLY, or OCaml/LLVM
- **No late submissions** — plan ahead!

<div class="highlight">

**This isn't your typical compiler course.** We'll connect every concept to modern AI systems.

</div>

> "Every AI system you will build is, at its core, a compiler pipeline."

---

## About Me & Teaching Style

- **Small class = personalized attention**
  - More discussion, less lecture
  - Student-led paper presentations
  - Hands-on GPU sessions when we cover AI compilers
  - Personalized project feedback

**My expectation:** You're here because you're genuinely interested in compilers, AI infrastructure, or both. Let's make this count.

---

## Prerequisites

I assume you have taken:

- **MAT107** Discrete Math
- **BIL 121** C Programming
- **BIL 201** Data Structures & Algorithms
- **BIL 206** Algorithm Design & Analysis

<div class="ai-callout">

**No AI/ML background required.** We'll introduce tensors, computational graphs, and AI concepts as needed, always connecting them back to compiler ideas you already know.

</div>

---

## Grading

| Component | Weight | Notes |
|-----------|--------|-------|
| Midterm | 20% | Similar to written assignments |
| Final exam | 40% | Written assignments + conceptual questions |
| Written assignments | 10% | 2-3 theoretical assignments |
| Programming projects | 30% | 3-4 milestones, groups of 2-3 |

### Policies
- **No late submissions** — plan your time
- **AI tools permitted** — but you must acknowledge usage and be able to explain every line you submit
- **Discuss with friends** — but submit your own work

---

## Textbooks & Materials

**No required textbook.** Weekly notes on Classroom.

### Primary References
- 📖 [Stanford CS143](https://web.stanford.edu/class/cs143/) — lecture notes heavily based on this
- 📖 [UW CSE401](https://courses.cs.washington.edu/courses/cse401/22au/)
- 📖 Dragon Book (Aho et al.)
- 📖 *Engineering a Compiler* — Cooper & Torczon, 3rd edition
- 📖 *Introduction to Compilers and Language Design* — Thain ([free online](https://www3.nd.edu/~dthain/compilerbook/))

---

### AI Compiler References (Supplementary)
- 🤖 Apache TVM: [tvm.apache.org](https://tvm.apache.org/)
- 🤖 MLIR: [mlir.llvm.org](https://mlir.llvm.org/)
- 🤖 Various survey papers (linked in lectures)

---

<!-- _class: lead -->

# Part 1
## Why Compilers in 2026?

---

## The Traditional Answer (Still True)

**Why learn compilers?**

- ✅ Understand and compare high-level languages better
- ✅ Write more efficient programs
- ✅ Design translating tools (transpilers, DSLs, etc.)
- ✅ Learn concepts used in many areas:
  - Parsing, regular expressions, ASTs, type systems
  - These appear in databases, browsers, security tools, bioinformatics

**These reasons remain valid.** But there's more...

---

## The New Answer 🚀

**Every AI system is a compiler pipeline.**

<div class="ai-callout">

Think about it: when you build an AI agent, what are you really doing?

</div>

---

| Traditional Compiler | AI / Agent System |
|---------------------|-------------------|
| Source code | User query (natural language) |
| Lexical analysis | LLM tokenization (BPE) |
| Parsing | Intent recognition / structured extraction |
| AST | Parsed intent representation |
| Semantic analysis | API argument validation |
| IR | Plan / chain-of-thought |
| Optimization | Plan optimization (reorder, fuse steps) |
| Code generation | Tool execution (API calls) |
| Runtime | Agent loop (observe, act, reflect) |
| Error recovery | Self-correction / retry logic |

---

## Real Examples: Compilers ARE AI Infrastructure

<div class="ai-callout">

### 📝 LLM tokenizers ARE lexers
BPE / SentencePiece are lexer generators for natural language. They take text → tokens, just like Flex takes text → tokens.

</div>

<div class="ai-callout">

### 📋 Structured output from LLMs IS parsing
Constrained decoding (Outlines, LMQL, JSON mode) = grammar-guided parsing. When you force an LLM to output valid JSON, you're doing parsing.

</div>

<div class="ai-callout">

### 🔍 Tensor shape checking IS semantic analysis
Every `RuntimeError: mat1 and mat2 shapes cannot be multiplied` is a type error. PyTorch's shape checking is semantic analysis.

</div>

---

## More Examples

<div class="ai-callout">

### 🧮 Computational graphs ARE IRs
ONNX, Relay IR, MLIR are intermediate representations — the "assembly language" of neural networks.

</div>

<div class="ai-callout">

### ⚡ Operator fusion IS peephole optimization
`Conv → BatchNorm → ReLU → FusedConvBNReLU` is the same idea as `X = Y * 0 → X = 0`.

</div>

<div class="ai-callout">

### 🎯 GPU kernel generation IS code generation
TVM/XLA generate CUDA/Metal kernels from computational graphs, just like compilers generate x86 from C.

</div>

---

## The Historical Pattern

```
1970s: C compiler enabled software revolution
       → Portable code across different CPUs

2000s: JIT compilers enabled web revolution  
       → JavaScript runs everywhere

2020s: AI compilers enabling AI revolution
       → Neural networks run everywhere
```

<div class="highlight">

**The skills in this course power TVM, XLA, MLIR, TensorRT.**

People who understand both compilers AND AI are rare and valuable.

</div>

---

## What Makes This Course Different

| Traditional Compiler Course | This Course |
|----------------------------|-------------|
| Focus only on C/Java compilers | Compilers for AI systems |
| Theory-heavy | Theory + modern applications |
| "Why do we still teach this?" | "Here's how this powers AI" |
| Large lecture | Small seminar + lab |
| Generic projects | AI-flavored project options |

**Same core content, modern context.**

---

<!-- _class: lead -->

# Part 2
## What Is a Compiler?

---

## The Question

How do we execute something like this?

```c
#include<stdio.h>
#define X 10
int main(){
    int a = X;
    printf("hello world!\n a=%d", a);
    return 0;
}
```

How do we tell a computer to carry out a computation written as **text in a file**?

---

## Definition

A **compiler** is a program that:

- Translates a program in a **source language** to a program in a **target language** (generally low-level)
- Also **improves** the program during translation (optimization)

```
Source program  ──→  [Compiler]  ──→  Target program
```

---

### Related Concepts
- **Cross compiler**: runs on machine A, produces code for machine B
- **Transpiler** (source-to-source compiler): translates source code into another high-level language
  - TypeScript → JavaScript
  - Kotlin → Java bytecode

---

## Two Implementation Strategies

### Interpreters run your program directly
- Read and execute without translation file
- Python, Ruby, sometimes via a VM
- Slower but more flexible

### Compilers translate your program
- C, C++, Go, Rust
- Faster execution, less flexible

### Some provide both
- Java, JavaScript, WebAssembly
- Interpreter + Just-In-Time (JIT) compiler

---

## Example: Python Bytecode

```python
>>> import dis  # "dis" - Disassembler of Python bytecode
>>> dis.dis('print("Hello, World!")')
  1    0 LOAD_NAME     0 (print)
       2 LOAD_CONST    0 ('Hello, World!')
       4 CALL_FUNCTION 1
       6 RETURN_VALUE
```

Python compiles to bytecode, then interprets it. This is a hybrid approach.

---

## Course Goals

**Open the lid of compilers and see inside:**

- ✅ Understand **what they do**
- ✅ Understand **how they work**
- ✅ Understand **how to build them**

<div class="warning">

**Correctness over performance** — a compiler must produce correct code first. Optimization comes later.

</div>

---

<!-- _class: lead -->

# Part 3
## The Structure of a Typical Compiler

---

## The 5 Phases

By analogy with how humans comprehend English:

1. **Lexical Analysis** → identify words
2. **Parsing** → identify sentences
3. **Semantic Analysis** → analyze meaning
4. **Optimization** → editing
5. **Code Generation** → translation

<div class="highlight">

**Front-end** (phases 1-3): Language-specific, deals with source code  
**Back-end** (phases 4-5): Target-specific, deals with machine code

</div>

---

## Phase 1: Lexical Analysis (Scanning)

Recognize **words** — the smallest unit above letters.

```
If x == y then z = 1; else z = 2;
```

↓ tokens ↓

```
IF  ID(x)  EQEQ  ID(y)  THEN  ID(z)  ASSIGN  NUMBER(1)
SEMICOLON  ELSE  ID(z)  ASSIGN  NUMBER(2)  SEMICOLON
```

<div class="ai-callout">

**AI connection:** This is exactly what an LLM tokenizer does. BPE takes text → tokens. The difference? BPE handles natural language ambiguity; Flex handles programming language precision.

</div>

---

## Lexical Analysis is Not Trivial

Consider:

```
ist his ase nte nce.
```

Where do the word boundaries go? The lexer must figure this out.

For programming languages, the rules are strict:
- Keywords: `if`, `while`, `return`
- Identifiers: `x`, `count`, `myVar`
- Operators: `+`, `-`, `*`, `/`, `==`
- Literals: `42`, `3.14`, `"hello"`

---

## Phase 2: Parsing

Groups tokens into **statements and expressions** — like diagramming sentences.

The diagram is a tree: the **Abstract Syntax Tree (AST)**.

```
If x == y then z = 1; else z = 2;

        If
       / | \
     ==   1   2
    /  \
   x    y
```

<div class="ai-callout">

**AI connection:** When an LLM produces JSON or tool calls, it's parsing. Constrained decoding = grammar-guided parsing. The LLM must follow a grammar to produce valid output.

</div>

---

## Phase 3: Semantic Analysis

Once structure is understood, we try to understand **meaning**.

Compilers perform **limited** semantic analysis to catch inconsistencies:

### Example 1: Scope
```c
{
    int Jack = 3;
    {
        int Jack = 4;
        printf("%d\n", Jack);  // Prints 4 — inner definition wins
    }
}
```

---

### Example 2: Type Checking
```
Jack left her homework at home.
```
If Jack is male → possible type mismatch between `her` and `Jack`.

---

## More Semantic Checks

Compilers check many things:

- ✅ All identifiers are declared
- ✅ Types are compatible
- ✅ Functions are called with correct arguments
- ✅ Classes defined only once
- ✅ Methods defined only once
- ✅ Reserved identifiers not misused

<div class="ai-callout">

**AI connection:** PyTorch shape checking = semantic analysis. When you see `RuntimeError: Expected dimension 3 but got 4`, that's a type error caught by semantic analysis.

</div>

---

## Phase 4: Optimization

Akin to editing — minimize reading time, minimize memory.

Automatically modify programs so they:
- Run faster
- Use less memory
- Conserve some resource (power, bandwidth, etc.)

---

### Example
```c
X = Y * 0    is the same as    X = 0
```

<div class="warning">

**Is this optimization always legal?** Think about:
- Floating-point NaN
- Side effects (what if Y is a function call?)
- Integer overflow

</div>

---

## Optimization in AI Compilers

<div class="ai-callout">

**Operator fusion** is peephole optimization:

```
Before: Conv → BatchNorm → ReLU → MaxPool
After:  FusedConvBNReLU → MaxPool
```

This is the same idea as `X = Y * 0 → X = 0`, but for neural networks.

</div>

Other AI compiler optimizations:
- Dead code elimination (remove unused layers)
- Constant folding (pre-compute static weights)
- Memory optimization (reuse buffers)

---

## Phase 5: Code Generation

Produces **assembly code** — analogous to human translation.

### Intermediate Representations (IRs)

Many compilers translate between successive intermediate languages:

```
Source (high-level)
    ↓
IR1 (medium-level)
    ↓
IR2 (low-level)
    ↓
Assembly (lowest-level)
```

---

## Why IRs?

**IRs are useful because:**

- Lower levels expose features hidden by higher levels
  - Registers, memory layout, raw pointers
- But lower levels obscure high-level meaning
  - Classes, higher-order functions, loops

**Trade-off:** Abstraction vs. control

<div class="ai-callout">

**AI connection:** ONNX, Relay IR, MLIR are IRs — the "assembly language" of neural networks. They allow optimization at different abstraction levels.

</div>

---

## An Example, Step by Step

```c
height = (width + 56) * factor(foo);
```

1. **Lexer** → tokens: `ID(height) ASSIGN LPAREN ID(width) PLUS NUMBER(56) ...`
2. **Parser** → builds AST from grammar rules
3. **Semantic routines** → traverse AST, derive meaning, check types
4. **Post-order traversal** → generates IR instructions
5. **Optimizer** → removes dead code, combines operations
6. **Codegen** → converts IR to x86 assembly

---

## Issues & Pitfalls

Compiling is almost this simple, but there are many pitfalls:

- ❓ How to handle erroneous programs?
  - Report errors and continue?
  - Stop immediately?
  - Try to recover and keep compiling?

- ❓ Language design has big impact on compiler
  - Determines what is easy and hard to compile
  - **Course theme:** many trade-offs in language design

---

## Compilers Today

The overall structure of almost every compiler adheres to our outline.

**But the proportions have changed since FORTRAN:**

- **Early:** lexing and parsing most complex/expensive
- **Today:** optimization dominates all other phases
  - Lexing and parsing are well understood and cheap
  - Optimization is where the magic happens

---

**Compilers are now also found inside libraries:**
- JIT compilers in databases
- Query optimizers
- Shader compilers in graphics

---

<!-- _class: lead -->

# Part 4
## Modern Compilers: LLVM

---

## What is LLVM?

**LLVM** = Low Level Virtual Machine

An open-source compiler infrastructure:

- **Core idea:** A language-agnostic **IR** at the heart of compilation
- **Key benefit:** Separates **front-end** (language-specific) from **back-end** (machine-specific)

```
Source code  →  [Frontend]  →  LLVM IR  →  [Optimizer]  →  LLVM IR  →  [Backend]  →  Machine code
```

---

## The LLVM Pipeline

### Three Major Components

1. **Frontend** (e.g., Clang for C/C++)
   - Translates source code to LLVM IR
   - Language-specific parsing, AST, semantic analysis

2. **Middle-end** (LLVM Optimizer)
   - Performs target-independent optimizations
   - `-O1`, `-O2`, `-O3` control optimization level
   - This is where most optimizations happen

3. **Backend** (LLVM Code Generator)
   - Converts optimized LLVM IR to native machine code
   - x86, ARM, RISC-V, etc.

---

## Why is LLVM a Big Deal?

### Reusability
- Create a new language by just writing a frontend that generates LLVM IR
- LLVM handles the hard part: optimization and code generation for multiple architectures

---

### Performance
- LLVM's optimizations are highly effective
- Continuously improved by a large community

### Industry Standard
- **Apple:** Swift, Clang
- **Google:** Android NDK, ML compilers
- **Rust:** rustc uses LLVM
- **Many others:** Julia, Swift, Clang

---

## Interesting Capabilities

### Link-Time Optimization (LTO)
- Optimize across translation units
- `gcc -O2 -flto program.c`

### Install-Time Optimization
- Optimize when software is installed, not when compiled

### Modular Design
- Mix and match frontends and backends
- Write a new frontend, reuse existing backends

---

## FP Concepts in Compiler Design

Modern compilers are inspired by functional programming:

---

### Immutability & Pure Functions
- Compiler passes take AST/IR → return new AST/IR
- No side effects → easier to reason about, test, debug

### Algebraic Data Types & Pattern Matching
- Perfect for ASTs:
  ```ocaml
  type expr =
    | BinaryExpr of op * expr * expr
    | Number of int
    | Variable of string
  ```

### Higher-Order Functions
- `map`, `fold` over lists of statements

---

## Languages Used in Modern Compilers

- **OCaml / Haskell** — research & production compilers
  - Original F# compiler, Glasgow Haskell Compiler
- **Rust** — safety + pattern matching
  - Excellent for writing compilers
- **Scala** — blends OOP and FP
  - Dotty/Scala 3 compiler

<div class="highlight">

**This is why we offer an OCaml path for your project.**

</div>

---

## Beyond the "Typical" Compiler

### Single-Pass Compilers
- Generate code immediately during parsing
- No full AST
- Use case: simple languages, memory-constrained environments

### Just-In-Time (JIT) Compilation
- Compile at runtime, right before execution
- Can optimize based on actual runtime data
- Examples: Java HotSpot, JavaScript V8, .NET CLR

---

### Interpreters & VMs
- **Tree-walk:** directly execute AST (simple but slow)
- **Bytecode:** compile to bytecode, then interpret (good balance)

### Transpilers
- Source-to-source compilation
- TypeScript → JavaScript
- Kotlin → Java bytecode

---

<!-- _class: lead -->

# Part 5
## Your Project: Choose Your Path

---

## Path A: C + Flex/Bison — The Classic

```
Source → [Flex] → Tokens → [Bison] → AST → [Your C] → Assembly
```

### What You'll Work With
- **Flex:** generates lexical analyzer from regex rules
- **Bison:** generates parser from grammar rules
- **Manual C code:** semantic analysis, code generation
- **Output:** x86 assembly or similar

---

### Example Code
```c
struct ASTNode {
    enum { NUMBER, BINOP, VARIABLE } type;
    union {
        int value;
        struct { char op; struct ASTNode *left, *right; } binop;
        char* varname;
    };
};
```

**Best for:** Understanding computer architecture, traditional approach.

---

## Path B: Python + PLY — Rapid Prototyping

```
Source → [PLY Lex] → Tokens → [PLY Yacc] → AST → [Your Python] → Bytecode
```

### What You'll Work With
- **PLY (Python Lex-Yacc):** pure Python lexing/parsing
- **Python classes:** AST, symbol tables, semantic analysis
- **Python eval** or simple stack-based bytecode generation
- **Output:** Python bytecode, custom VM, or simple assembly

---

### Example Code
```python
class BinOp:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

def generate_code(node):
    if isinstance(node, BinOp):
        generate_code(node.left)
        generate_code(node.right)
        print(f"{op_to_asm[node.op]}")  # e.g., "add"
```

**Best for:** Focus on algorithms over systems details, easier debugging.

---

## Path C: OCaml + LLVM — Industry-Inspired

```
Source → [OCaml Lex] → Tokens → [OCaml Parse] → AST → [LLVM IR] → Native Code
```

### What You'll Work With
- **OCaml Lex/Yacc** or parser combinators (Menhir)
- **Algebraic data types:** perfect for AST representation
- **LLVM bindings:** professional optimization and code generation
- **Pattern matching:** elegant AST traversal and transformation

---

### Example Code
```ocaml
type expr =
  | Number of int
  | BinOp of string * expr * expr
  | Variable of string

let rec codegen expr builder =
  match expr with
  | Number n -> Llvm.const_int i32_type n
  | BinOp ("+", lhs, rhs) ->
      let l = codegen lhs builder in
      let r = codegen rhs builder in
      Llvm.build_add l r "addtmp" builder
```

**Best for:** Modern compiler architecture + functional programming.

---

## How to Choose?

| Question | Path A | Path B | Path C |
|----------|--------|--------|--------|
| Strong C background? | ✅ | | |
| Python comfort? | | ✅ | |
| Want FP challenge? | | | ✅ |
| Understand architecture? | ✅ | | |
| Focus on theory? | | ✅ | |
| Learn industry tools? | | | ✅ |
| Risk tolerance | Traditional | Safe | Ambitious |

<div class="highlight">

**All paths teach the same concepts.** The concepts are the same; only the implementation differs. You can switch paths early if needed.

</div>

---

## Project Expectations

### Common Requirements (All Paths)
- Implement MiniLang with variables, functions, arithmetic
- Handle lexical errors, syntax errors, type errors
- Produce executable output
- Write tests and documentation

---

### Milestones
1. **Milestone 1:** Lexer
2. **Milestone 2:** Parser + AST
3. **Milestone 3:** Semantic analysis & type checking
4. **Milestone 4:** IR + code generation

---

## AI-Flavored Extensions (Optional Bonus)

Pick one as a bonus on your project:

### 1. 🧮 Compile a Tiny Tensor DSL
```
tensor C = matmul(A, B);
relu(C);
```
→ Generate fused NumPy or C code

---

### 2. 🤖 Build a Mini "Agent Compiler"
Task DSL → execution plan
```
fetch weather for Istanbul
email result to ali@x.com
```

---

### 3. 💬 Write a Prompt Optimizer
Apply classic compiler passes to LLM prompts:
- Dead-code elimination
- Constant folding
- Fusion

---

### 4. 🔍 Use an LLM to Write Your Compiler
Critically analyze where it succeeds and fails

---

## Environment Setup

You need a **Linux machine or VM**.

```bash
# Path A: C / Flex / Bison
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install build-essential flex bison

# Path B: Python / PLY
pip install ply

# Path C: OCaml / LLVM
# See setup guide linked in lecture 1
```

Windows users: use [WSL](https://learn.microsoft.com/en-us/windows/wsl/).

---

## Academic Integrity

### What's Allowed
- ✅ Discussing assignments/projects with friends
- ✅ Using AI tools (ChatGPT, Copilot, DeepSeek, etc.)

### What's Required
- ✅ All submitted work must be **yours**
- ✅ List collaborators if you discussed with friends
- ✅ **Acknowledge AI tool usage** in your reports
- ✅ Be prepared to **explain every line** you submit

---

### What's Prohibited
- ❌ Plagiarism and cheating (see university policy)
- ❌ Submitting work