# BIL 463 Compiler Design: From Foundations to AI Systems (Fall 2026)

**Lecture:** 9:30 on Tuesdays 
**Instructor:** Ammar Daşkın
**Semester:** Fall 2026

---

## Course Overview

This course is an undergraduate-level introduction to the design principles of compilers and interpreters, with a modern emphasis on how compiler technology powers today's AI systems. **Every AI system you will build is, at its core, a compiler pipeline** — from LLM tokenizers (lexical analysis) to grammar-guided generation (parsing), from computational graphs (IR) to GPU kernel generation (code generation), and from agent frameworks to compiler-style optimization passes.

We will build a mini-compiler for a small language as a semester-long programming project, while weaving in modern AI applications at every stage. By the end of the course, you will understand not only how traditional compilers work, but also how the same concepts appear in PyTorch, TVM, MLIR, LLM tokenizers, and AI agent systems.

### Why This Course Matters in 2026

- **LLM tokenizers are lexers** — BPE/SentencePiece are lexer generators for natural language
- **Structured output from LLMs is parsing** — Constrained decoding (Outlines, LMQL, JSON mode) is grammar-guided parsing
- **Tensor shape checking is semantic analysis** — PyTorch shape errors are type errors
- **Computational graphs are IRs** — ONNX, Relay IR, and MLIR are intermediate representations
- **Agent frameworks are compiler pipelines** — Plan generation → optimization → tool execution mirrors IR → optimization → codegen

---

## Prerequisites

You are expected to have prior C programming experience and sufficient knowledge of data structures and algorithms. I will assume you have taken at least the following courses:

- MAT107 Discrete Math
- BIL 121 C Programming
- BIL 201 Data Structures and Intro to Algorithms
- BIL 206 Algorithm Design and Analysis

**No prior AI/ML background is required.** We will introduce tensor operations, computational graphs, and AI concepts as needed, always connecting them back to compiler ideas you already know.

---

## Textbooks and Course Material

No required textbook. Lecture notes are posted weekly on Classroom.

**Primary references:**
- *Introduction to Compilers and Language Design*, Douglas Thain, 2nd edition, 2020. [Free online](https://www3.nd.edu/~dthain/compilerbook/)
- [Stanford CS143](https://web.stanford.edu/class/cs143/) — lecture notes heavily based on this
- [UW CSE401](https://courses.cs.washington.edu/courses/cse401/22au/)
- *Compilers: Principles, Techniques, & Tools* (Dragon Book), Aho, Lam, Sethi & Ullman
- *Engineering a Compiler*, Cooper & Torczon, 3rd edition

**AI compiler references (supplementary):**
- Apache TVM documentation: [tvm.apache.org](https://tvm.apache.org/)
- MLIR documentation: [mlir.llvm.org](https://mlir.llvm.org/)
- *Engineering a Compiler for Machine Learning* (various survey papers, linked in lectures)

---

## Weekly Schedule

| Week | Topic | AI / Modern Angle |
|------|-------|-------------------|
| 1  | Administrivia & Introduction | **Why compilers = AI infrastructure.** Overview of the AI compilation pipeline. LLM tokenization as lexical analysis. |
| 2  | Lexical Analysis I — Regular expressions, finite automata | LLM tokenizers (BPE, SentencePiece) as lexer generators. Why regex fails for natural language. |
| 3  | Implementation of Lexical Analysis (Flex / PLY) | Hands-on: comparing a Flex lexer with a BPE tokenizer on the same input. |
| 4  | Introduction to Parsing — CFGs, derivations | Grammar-guided LLM generation (Outlines, LMQL). "Making an LLM output valid JSON" is parsing. |
| 5  | Top-Down Parsing (LL, recursive descent) | Parsing agent tool-call outputs. Structured extraction from LLM responses. |
| 6  | Bottom-Up Parsing (LR, SLR, LALR) | Syntax-directed translation and its role in modern DSLs. |
| 7  | Semantic Analysis & Type Checking | **Tensor shape checking as semantic analysis.** Real PyTorch shape errors mapped to type errors. Symbol tables for typed AI pipelines. |
| 8  | Run-time Environments | **Agent runtime systems.** Memory management for context windows, tool execution sandboxes, conversation state. |
| 9  | Code Generation | **Compiling for AI hardware.** GPU kernel generation, what TVM/XLA actually produce. LLMs as code generators. |
| 10 | Intermediate Representations & Local Optimization | **Computational graphs as IR.** Operator fusion (Conv+BN+ReLU) as peephole optimization. |
| 11 | Global Optimization | **ML for compiler optimization.** AutoTVM, learned heuristics, neural program synthesis. |
| 12 | Instruction Scheduling & Register Allocation | **Tensor memory planning.** Activation checkpointing and memory-efficient attention as register allocation problems. |
| 13 | AI Compilers Deep Dive | TVM, MLIR, XLA, TensorRT. The full AI compilation pipeline. |
| 14 | Agents as Compiler Pipelines | The analogy: user query → tokenization → parsing → plan IR → optimization → tool execution → runtime. |
| 15 | Project Presentations & Semester Summary | Modern trends, open problems, and where to go next. |

---

## Homework & Programming Assignments

### Programming Projects (30%)
You will design and implement a compiler for a small language, in 3–4 incremental milestones. The project stops at code generation (no full optimization pass required), but you are encouraged to add **one AI-related extension** as a bonus:

- **Milestone 1:** Lexer
- **Milestone 2:** Parser + AST
- **Milestone 3:** Semantic analysis & type checking
- **Milestone 4:** IR + code generation

**Suggested AI-flavored extensions (pick one, optional but encouraged):**
1. Compile a tiny tensor DSL (e.g., `tensor C = matmul(A, B); relu(C);`) to fused NumPy or C.
2. Build a mini "agent compiler" that takes a task DSL and produces an execution plan.
3. Write a prompt optimizer that applies classic compiler passes (dead-code elimination, constant folding, fusion) to LLM prompts.
4. Use an LLM to help write your compiler, then critically analyze where it succeeds and fails.

You may work in groups of 2 or 3. Submissions through Classroom and GitHub. **No late submissions.**

### Written Assignments (10%)
2–3 theoretical assignments on automata, grammars, parsing tables, type systems. You may use AI tools, but you must acknowledge usage in your report.

### Exams
- **Midterm (20%):** Similar in style to written assignments.
- **Final (40%):** Written assignments + conceptual questions related to your project and the AI/compilers connection.

---

## Grading Summary

| Component | Weight |
|-----------|--------|
| Midterm | 20% |
| Final exam | 40% |
| Written assignments | 10% |
| Programming projects | 30% |

---

## Implementation Paths for the Project

You may choose one of three implementation paths — all build the same language, with different tools and trade-offs:

- **Path A: C + Flex/Bison** — The classic approach. Understand everything from scratch.
- **Path B: Python + PLY** — Rapid prototyping. Focus on concepts over systems details.
- **Path C: OCaml + LLVM** — Industry-inspired. Leverage functional programming and a real optimization/codegen backend.

All paths will teach you compiler design. The concepts are the same; only the implementation differs.

---

## Collaboration and Academic Integrity

- Any kind of plagiarism and cheating is prohibited. Please refer to the university cheating policy.
- Discussing assignments and projects with friends is allowed, but **all submitted work must be your own**.
- If you benefit from work of others (including AI tools), list them as references.
- **Use of AI tools (ChatGPT, Copilot, DeepSeek, etc.) is permitted** for assignments and projects, but you must acknowledge this in your reports and be prepared to explain every line you submit.

---

## Environment

You should have a Linux machine or VM. Required tools depend on your chosen path:

```bash
# For Path A (C/Flex/Bison)
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install build-essential flex bison

# For Path B (Python/PLY)
pip install ply

# For Path C (OCaml/LLVM)
# See OCaml and LLVM setup guides linked in lecture 1
```

Windows users can use [WSL](https://learn.microsoft.com/en-us/windows/wsl/).

---

## Discussions & Questions

We use **Classroom (Google Classroom)** for assignment submission, grading, and discussions.

- Do not post solutions or significant parts of an assignment.
- Do not post anything unrelated to the course.
- Ask when you need help; answer when you can help others.

---

## A Note on the Small Class

With a small cohort this semester, we will run the course more like a **seminar + lab** than a large lecture. Expect:

- More discussion and Q&A during lectures
- Student-led paper presentations on AI compiler topics (TVM paper, MLIR, ReAct agents, etc.)
- Hands-on GPU sessions when we cover AI compilers
- Personalized feedback on projects

If you are interested in compilers, AI infrastructure, or both — this course is built for you.

---

## Resources & Community

- **Conferences:** PLDI, CGO, MLSys, LLVM Developers' Meeting
- **Open-source projects to explore:** LLVM/Clang, Apache TVM, MLIR, ONNX Runtime, V8, Roslyn
- **Communities:** CompilerDev Discord, `#llvm` on IRC, r/compilers

---

