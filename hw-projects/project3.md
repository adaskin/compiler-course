# **Project 3: Semantic Analysis and Code Generation**
*written/prepared with the help of DeepSeek AI*

---

- [**Project 3: Semantic Analysis and Code Generation**](#project-3-semantic-analysis-and-code-generation)
  - [**1. Objective**](#1-objective)
  - [**2. Learning Outcomes**](#2-learning-outcomes)
  - [**3. Project Requirements**](#3-project-requirements)
    - [**3.1 Semantic Analysis Phase**](#31-semantic-analysis-phase)
    - [**3.2 Code Generation Phase**](#32-code-generation-phase)
  - [**4. Starter Code Reference**](#4-starter-code-reference)
    - [**4.1 Semantic Analysis Examples**](#41-semantic-analysis-examples)
    - [**4.2 Code Generation Examples**](#42-code-generation-examples)
    - [**4.3 Project 2 Extensions**](#43-project-2-extensions)
  - [**5. Implementation Guidelines**](#5-implementation-guidelines)
    - [**5.1 Build on Your Project 2**](#51-build-on-your-project-2)
    - [**5.2 Recommended Implementation Path**](#52-recommended-implementation-path)
    - [**5.3 Example Architecture**](#53-example-architecture)
  - [**6. Test Programs**](#6-test-programs)
    - [**6.1 Semantic Analysis Tests**](#61-semantic-analysis-tests)
    - [**6.2 Code Generation Tests**](#62-code-generation-tests)
    - [**6.3 Integration Tests**](#63-integration-tests)
  - [**7. Deliverables**](#7-deliverables)
    - [**7.1 Source Code**](#71-source-code)
    - [**7.2 Updated README.md**](#72-updated-readmemd)
    - [**7.3 Test Suite**](#73-test-suite)
    - [**7.4 Sample Output**](#74-sample-output)
  - [**8. Grading Rubric**](#8-grading-rubric)
  - [**9. Reference Implementation Hints**](#9-reference-implementation-hints)
    - [**9.1 Building on Provided Examples**](#91-building-on-provided-examples)
    - [**9.2 Recommended Development Order**](#92-recommended-development-order)
  - [**10 Common Issues \& Solutions**](#10-common-issues--solutions)
    - [**Resources**](#resources)
  - [**Policy Reminders**](#policy-reminders)

## **1. Objective**

Building on your successful parser and AST from Project 2, you will now implement the **semantic analysis** and **code generation** phases of your compiler. 
- This project goes from program structure (AST) to producing executable code.
**- Feel free to change your programming language specifics you have used in project2.**
  

**Semantic Analysis:** Your compiler will verify that programs follow the language's semantic rules (type checking, scope rules, proper variable usage) that couldn't be captured by context-free grammar alone.

**Code Generation:** Your compiler will translate the validated AST into executable code, either for a real architecture (MIPS/x86) or a virtual machine/bytecode format.

---

## **2. Learning Outcomes**

By completing this project, you will:
- Understand the difference between syntax (structure) and semantics (meaning)
- Implement type checking and scope analysis
- Design and generate intermediate or target code
- Connect AST representation to executable instructions
- Handle error recovery and reporting in semantic analysis

---

## **3. Project Requirements**
**- Feel free to change your programming language specifics you have used in project2.**
However, you should at least have the basic minimum here in project3 as required in project2
 
### **3.1 Semantic Analysis Phase**

Your compiler must implement:
- **Type checking**: Verify type compatibility in expressions, assignments, and function calls
- **Variable resolution**: Ensure all variables are declared before use
- **Scope analysis**: Handle nested scopes and shadowing correctly
- **Function validation**: Check function signatures and return types
- **Error reporting**: Provide clear, informative error messages with line numbers

### **3.2 Code Generation Phase**

Choose **ONE** of these output formats:

**Option A: Stack-Based Bytecode** (Easier)
- Design a simple virtual machine instruction set
- Generate bytecode from AST
- Implement a stack-based interpreter

**Option B: MIPS Assembly** (Intermediate)
- Generate MIPS assembly code
- Handle stack frames for function calls
- Implement basic register allocation

**Option C: LLVM IR** (Advanced)
- Generate LLVM Intermediate Representation
- Leverage LLVM's optimization and code generation
- Produce native executables via LLVM

---

## **4. Starter Code Reference**

You have several examples to build upon:

### **4.1 Semantic Analysis Examples**
Reference the **8-semantic-analysis** lecture notes and code examples. These demonstrate:
- Symbol table implementation with scoping
- Type checking for expressions and assignments
- Error recovery during semantic analysis

**Key Concepts from Reference:**
```c
// Example type checking pattern
Type* check_binary_expression(Type* left, Type* right, Operator op, int line) {
    if (!types_compatible(left, right, op)) {
        report_error(line, "Type mismatch in expression");
        return ERROR_TYPE;
    }
    return result_type(left, right, op);
}
```

### **4.2 Code Generation Examples**
Reference the **5-bison-example** directory, specifically:
- **`codegen.h`** - Bytecode instruction definitions
- **`codegen.c`** - Implementation of code generation and execution

**Example Pattern:**
```c
// From 5-bison-example/codegen.c
Bytecode* generate_code(ASTNode* ast) {
    Bytecode* bc = create_bytecode();
    traverse_ast(ast, bc);  // Generate instructions
    return bc;
}
```

### **4.3 Project 2 Extensions**
Your existing Project 2 code provides:
- AST structure (`ast.h`, `ast.c`)
- Parser with basic type annotations
- From examples 8: Symbol table foundation (`symbol.h`, `symbol.c`)

---

## **5. Implementation Guidelines**

### **5.1 Build on Your Project 2**
Start with your working Project 2 codebase. You should:
1. **Extend your AST** to include type information at each node
2. **Enhance your symbol table** to track types and scopes
3. **Add semantic analysis passes** that traverse the AST
4. **Implement code generation** as a final pass

### **5.2 Recommended Implementation Path**

#### **Phase 1: Semantic Analysis (Week 1-2)**
1. **Extend AST with types**: Add type field to all AST nodes
2. **Implement type checker**: Create `typecheck.c` with visitor pattern
3. **Add scope management**: Extend symbol table for nested scopes
4. **Write semantic tests**: Create both valid and invalid programs

#### **Phase 2: Code Generation (Week 3-4)**
1. **Choose target**: Bytecode, MIPS, or LLVM IR
2. **Design instruction set**: If choosing bytecode
3. **Implement code generator**: Create `codegen.c` with AST traversal
4. **Add execution/runtime**: Either interpreter or assembler/linker

### **5.3 Example Architecture**
```
           +---------------+
           |   Source Code |
           +---------------+
                   |
                   v
           +---------------+
           |     Parser    |   (from Project 2)
           +---------------+
                   |
                   v
           +---------------+
           |      AST      |   (extended with types)
           +---------------+
                   |
         +---------+---------+
         |                   |
         v                   v
+----------------+   +----------------+
| Type Checking  |   | Symbol Table   |
| & Validation   |   | Management     |
+----------------+   +----------------+
         |
         v
+----------------+
| Code Generator |  → MIPS/Bytecode/LLVM
+----------------+
```

---

## **6. Test Programs**

You must provide test cases that demonstrate:

### **6.1 Semantic Analysis Tests**
```c
// test_semantic_valid.ml (should pass)
int x = 5;
float y = 3.14;
bool flag = true;

// test_semantic_invalid.ml (should fail with clear errors)
int x = "hello";      // Type mismatch
y = 10;              // Undeclared variable
int x = 5;           // Redeclaration in same scope
```

### **6.2 Code Generation Tests**
```c
// test_codegen.ml
int main() {
    int x = 5;
    int y = x * 2 + 3;
    return y;  // Should generate code that returns 13
}
```

### **6.3 Integration Tests**
Create programs that combine:
- Multiple variable declarations
- Arithmetic expressions
- Control flow (if/while)
- Function calls (if your language supports them)

---

## **7. Deliverables**

### **7.1 Source Code**
- All source files with clear comments
- Updated build system (Makefile/CMake/etc.)
- **NOTE**: You must submit code that builds and runs

### **7.2 Updated README.md**
Your README must include:
1. **Build Instructions**: How to compile and run your compiler
2. **Language Specification**: Complete description of your language's semantics
3. **Implementation Details**: 
   - Semantic analysis approach
   - Target architecture choice and justification
   - Any limitations or known issues
4. **Test Instructions**: How to run the test suite

### **7.3 Test Suite**
- **10-15 test programs** covering various language features
- Each test should have expected output
- Include both valid and invalid programs
- Test output format (choose one):
  - Assembly/bytecode files
  - Execution results
  - Error messages

### **7.4 Sample Output**
For a simple program like `int x = 5 + 3;`, your compiler should produce:

**If generating MIPS:**
```mips
# Generated MIPS assembly
main:
    li $t0, 5
    li $t1, 3
    add $t2, $t0, $t1
    sw $t2, -4($fp)  # Store in x
```

**If generating bytecode:**
```
# Generated bytecode
PUSH 5
PUSH 3
ADD
STORE x
```

---

## **8. Grading Rubric**

| Category | Points | Criteria |
|----------|--------|----------|
| **Semantic Analysis** | 40 | - Complete type checking<br>- Scope resolution<br>- Clear error messages<br>- Handles edge cases |
| **Code Generation** | 35 | - Correct code generation<br>- Working output format<br>- Handles language features<br>- Clean generated code |
| **Test Suite** | 15 | - Comprehensive tests<br>- Clear expected results<br>- Covers edge cases |
| **Code Quality & Documentation** | 10 | - Clean, well-commented code<br>- Clear README<br>- Proper build system |

**Total: 100 points- Presentation: Mandatory(submissions not presented in the class will not be graded)**

---

## **9. Reference Implementation Hints**

### **9.1 Building on Provided Examples**

**Extend the AST from Project 2:**
```c
// In ast.h, add type field if not present
struct ASTNode {
    NodeType type;
    Type* node_type;  // Add this field
    // ... existing fields
};
```

**Add Semantic Analysis Pass:**
```c
// semanalysis.c
void analyze_program(ASTNode* program, SymbolTable* symtab) {
    // First pass: collect declarations
    collect_declarations(program, symtab);
    
    // Second pass: type checking
    check_types(program, symtab);
    
    // Third pass: other semantic checks
    validate_semantics(program, symtab);
}
```

**Integrate Code Generation:**
```c
// main.c (updated from Project 2)
int main() {
    // Parse (Project 2)
    ASTNode* ast = parse_file("program.ml");
    
    // Semantic analysis (Project 3)
    SymbolTable* symtab = create_symbol_table();
    if (!analyze_program(ast, symtab)) {
        exit(1);  // Semantic errors found
    }
    
    // Code generation (Project 3)
    if (output_format == "mips") {
        generate_mips(ast, "output.s");
    } else if (output_format == "bytecode") {
        Bytecode* bc = generate_bytecode(ast);
        save_bytecode(bc, "output.bc");
    }
    
    return 0;
}
```

### **9.2 Recommended Development Order**
1. Start with simple expressions (arithmetic)
2. Add variable declarations and assignments
3. Implement control structures
4. Add function support (if in your language)
5. Test incrementally after each feature

- **Week 1**: 
  - Implement basic type checking and extend AST
  - Complete semantic analysis with scope management
- **Week 2**: 
  - Implement code generation for simple expressions
  -  Add control flow, test thoroughly, document

---

## **10 Common Issues & Solutions**
- **Type checking is complex**: Start with simple types (int, bool), add others gradually
- **Code generation overwhelming**: Begin with stack-based bytecode, it's easier to debug
- **Scope management confusing**: Use the symbol table from Project 2 as a foundation

### **Resources**
- **Lecture notes**: 8-semantic-analysis, 10-code-generation
- **Example code**: 5-bison-example directory
- **Reference books**: "Engineering a Compiler" sections on semantic analysis and code generation

---

## **Policy Reminders**

- **Collaboration**: You may discuss approaches but must write your own code
- **AI Tools**: Permitted for brainstorming and debugging, but you must understand all submitted code
- **Language Evolution**: You may refine your language specification, but document changes
- **Late Policy**: No late submission

---

**Due Date: Last week-before presentation**
**Submission: Via Classroom-Presentation at the last lecture**

*Good luck, and remember: Compiler development is incremental. Build one feature at a time, test thoroughly, and don't be afraid to refactor as you learn more about your language's requirements!*