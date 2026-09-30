## **Code Generation - From AST to Machine Code**

### **The Final Compiler Phase**

```
Frontend Complete:           Backend:
AST + Symbol Table    →    Target Code
Type Information      →    Executable Program
Semantic Checks       →    Runtime Efficiency
```

### **Key Challenge:**
**How do we translate our high-level program understanding into low-level machine instructions?**

---

## **Intermediate Representations**
- Graphical IRs 
  - syntax related trees: AST, Parse Tree
  - Graphs: control-flow graph, dependency graph (DAG), call graph etc
- Linear IRs
  - stack machine code
  - one address, two-address, three address code (t5 ← t4 - t3)

---

## **Code Generation Strategies**
### **Three Major Approaches**

**1. Stack-Based (Interpreters/Virtual Machines)**
```java
// Java Bytecode-like
push 5
push 3
add
store result
```

---

**2. Register-Based (Traditional Compilers)**
```assembly
# x86-64
mov rax, 5
add rax, 3
mov [result], rax
```

---

**3. Modern SSA Form (LLVM, GCC)**
```llvm
; LLVM IR
%1 = add i32 5, 3
store i32 %1, i32* @result
```
---


### **Why Start with Stack Machines?**
- **Simpler** to understand and implement
- **Uniform** code generation pattern
- **Educational** foundation for more complex approaches

---

## **Stack Machine Fundamentals**

### **The Simple Evaluation Model**

**Basic Operations:**
- `push value` - push value onto stack
- `pop` - remove top value
- `add` - pop two values, push their sum
- `sub` - pop two values, push their difference

---

**Example: `(7 + 5) * 2`**
```
push 7      [7]
push 5      [7, 5]
add         [12]
push 2      [12, 2]
multiply    [24]
```

---

### **Real-World Usage:**
- **Java Bytecode** (JVM)
- **WebAssembly**
- **Python Bytecode**
- **.NET CIL**

---

## **From Stack Machine to Real Hardware**

---

## **Introduction to MIPS Architecture**

### **Why MIPS for Teaching Code Generation?**

**MIPS (Microprocessor without Interlocked Pipeline Stages)**
- **Designed for education**: Clean, regular instruction set
- **RISC architecture**: Reduced Instruction Set Computer
- **Load-store model**: Clear separation between memory and ALU operations
- **Simple addressing modes**: Easy to understand and generate code for

---

### **MIPS vs x86 NASM Comparison**

| Aspect | **MIPS (Our Focus)** | **x86 NASM (Common Alternative)** |
|--------|----------------------|-----------------------------------|
| **Architecture** | RISC | CISC |
| **Registers** | 32 general-purpose ($0-$31) | Fewer general registers (EAX, EBX, etc.) |
| **Instruction Format** | Fixed 32-bit, regular | Variable length, complex |
| **Memory Access** | Load/Store only (separate instructions) | Memory operands in most instructions |
| **Calling Convention** | Clear register usage ($a0-$a3 for args, $v0-$v1 for return) | Stack-based with some register args |
| **Teaching Value** | **Excellent** - simple, predictable | Complex, historical quirks |

---

### **MIPS Registers You Need to Know:**
```mips
# Important MIPS registers for our compiler
$a0-$a3:    # Argument registers (first 4 function args)
$v0-$v1:    # Return value registers
$t0-$t9:    # Temporary registers (caller-saved)
$s0-$s7:    # Saved registers (callee-saved)
$gp:        # Global pointer
$sp:        # Stack pointer
$fp:        # Frame pointer
$ra:        # Return address
```
[mips summary](https://www.cs.tufts.edu/comp/140/lectures/Day_3/mips_summary.pdf)  

---

### **MIPS Instruction Examples:**
```mips
# Load/Store
lw $t0, 4($sp)      # Load word from memory (stack+4) into $t0
sw $a0, 0($sp)      # Store word from $a0 to memory (stack)

# Arithmetic
add $a0, $t1, $t2   # $a0 = $t1 + $t2
sub $a0, $t1, $t2   # $a0 = $t1 - $t2
mul $a0, $t1, $t2   # $a0 = $t1 * $t2

# Control flow
beq $t0, $t1, label # Branch if equal
j label             # Unconditional jump
jal func            # Jump and link (call function)
jr $ra              # Jump to return address

# Immediate operations
addi $t0, $t1, 5    # $t0 = $t1 + 5
li $t0, 42          # Load immediate: $t0 = 42
```

---


### **Example: Simple Expression `a + b * c`**

**AST Representation:**
```
      +
     / \
    a   *
       / \
      b   c
```
**MIPS Code Generation:**
```mips
# Assuming: a in $s0, b in $s1, c in $s2
# Result in $a0

# Compute b * c
mul $t0, $s1, $s2   # $t0 = b * c

# Compute a + (b * c)
add $a0, $s0, $t0   # $a0 = a + $t0
```
---



**Stack Machine Approach `a + b * c` (for comparison):**
```
push a      # [a]
push b      # [a, b]
push c      # [a, b, c]
multiply    # [a, (b*c)]
add         # [a+(b*c)]
```

---

### **Simulating Stack on MIPS**

---
**Stack Machine:** `push 7`
```mips
li $a0, 7       # Load 7 into accumulator
sw $a0, 0($sp)  # Store accumulator to stack
addiu $sp, $sp, -4  # Move stack pointer down
```

**Register Mapping:**
- **Accumulator**: `$a0` - holds current computation result
- **Stack Pointer**: `$sp` - points to top of stack
- **Temporary**: `$t1` - for intermediate operations
- **Frame Pointer**: `$fp` - for stable variable access

---

 **Example: Variable Assignment `x = y + 5`**

**AST:**
```
    =
   / \
  x   +
     / \
    y   5
```

**MIPS Code Generation with Stack:`x = y + 5`**
```mips
# Assuming y's value is in memory at offset 8 from $fp

# Load y
lw $a0, 8($fp)      # $a0 = y
sw $a0, 0($sp)      # Push y onto stack
addiu $sp, $sp, -4

# Load 5
li $a0, 5           # $a0 = 5

# Add
lw $t1, 4($sp)      # Pop y into $t1
add $a0, $t1, $a0   # $a0 = y + 5
addiu $sp, $sp, 4   # Adjust stack

# Store result into x (offset 4 from $fp)
sw $a0, 4($fp)      # x = $a0
```

---


## **Code Generation for Expressions**

### **Recursive AST Traversal**

```python
def cgen(node):
    if node.type == 'number':
        return f"li $a0, {node.value}"
    
    elif node.type == 'binary_op':
        left_code = cgen(node.left)
        right_code = cgen(node.right)
        
        if node.operator == '+':
            return f"""
            {left_code}
            sw $a0, 0($sp)
            addiu $sp, $sp, -4
            {right_code}
            lw $t1, 4($sp)
            add $a0, $t1, $a0
            addiu $sp, $sp, 4
            """
```

### **Example: `3 + (7 + 5)`**
```mips
# 3 + (7 + 5)
li $a0, 3
sw $a0, 0($sp)
addiu $sp, $sp, -4

li $a0, 7
sw $a0, 0($sp)
addiu $sp, $sp, -4

li $a0, 5
lw $t1, 4($sp)
add $a0, $t1, $a0
addiu $sp, $sp, 4

lw $t1, 4($sp)
add $a0, $t1, $a0
addiu $sp, $sp, 4
```

---

## **Control Flow Generation**

### **Conditionals and Loops**

**If-Then-Else Pattern:**
```mips
cgen(condition)
beq $a0, $zero, false_branch

true_branch:
cgen(true_expression)
b end_if

false_branch:
cgen(false_expression)

end_if:
# Continue execution...
```

### **Example 3: If-Statement**
```c
// C code
if (x > 0) {
    y = 10;
} else {
    y = 20;
}
```
**MIPS Code Generation:**
```mips
# Assume x is at offset 8($fp), y at offset 12($fp)

# Load x
lw $t0, 8($fp)      # $t0 = x

# Compare x > 0
sgt $t1, $t0, $zero # $t1 = (x > 0) ? 1 : 0
beq $t1, $zero, else_label  # Branch if false

# Then part: y = 10
li $t2, 10
sw $t2, 12($fp)     # y = 10
j end_if            # Skip else part

else_label:
# Else part: y = 20
li $t2, 20
sw $t2, 12($fp)     # y = 20

end_if:
# Continue with next statements...
```

---

**While Loop Pattern:**
```mips
loop_start:
cgen(condition)
beq $a0, $zero, loop_end

cgen(loop_body)
b loop_start

loop_end:
# Continue after loop
```

---

## **Function Call Implementation**

### **The Calling Convention**

**Caller Responsibilities:**
1. Save frame pointer
2. Push arguments (reverse order)
3. Save return address
4. Jump to function

**Callee Responsibilities:**
1. Set up new frame pointer
2. Execute function body
3. Restore previous frame
4. Return to caller
---

```mips
# Caller: preparing f(x, y)
sw $fp, 0($sp)       # Save frame pointer
addiu $sp, $sp, -4

cgen(y)              # Push second argument
sw $a0, 0($sp)
addiu $sp, $sp, -4

cgen(x)              # Push first argument  
sw $a0, 0($sp)
addiu $sp, $sp, -4

jal f_entry          # Jump to function
```

---

## **Activation Records Revisited**

---

### **Connecting Runtime with Code Generation**

**Complete Activation Record for `f(x, y)`:**

```
High Address
┌─────────────┐
│   Old FP    │ ← $fp points here
├─────────────┤
│  Return Addr│
├─────────────┤
│      y      │ ← $fp + 8
├─────────────┤
│      x      │ ← $fp + 4
├─────────────┤
│  Temp NT(e) │
├─────────────┤
│     ...     │
├─────────────┤
│   Temp 1    │
└─────────────┘ ← $sp
Low Address
```
---

### **Variable Access:**
```mips
# Access parameter x (offset +4 from $fp)
lw $a0, 4($fp)

# Access parameter y (offset +8 from $fp)  
lw $a0, 8($fp)
```

---

## **Object-Oriented Code Generation**

### **From Classes to Machine Structures**

**Class Definition:**
```java
class Point {
    int x, y;
    void move(int dx, int dy) {
        x = x + dx;
        y = y + dy;
    }
}
```
---

**Memory Layout:**
```
Point Object:
┌─────────────┐
│  Class Tag  │  // Runtime type info
├─────────────┤
│ Object Size │  // For garbage collection  
├─────────────┤
│ Dispatch Ptr│  // Virtual method table
├─────────────┤
│      x      │  // Instance variable
├─────────────┤
│      y      │  // Instance variable
└─────────────┘
```

---

## **Dynamic Dispatch Implementation**

### **Virtual Method Tables**

**Class Hierarchy:**
```
     Animal
     /    \
   Dog    Cat
```
---

**Virtual Method Tables:**
```
Animal VTable    Dog VTable      Cat VTable
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  speak()    │  │  speak()    │  │  speak()    │
│  eat()      │  │  eat()      │  │  eat()      │
│  sleep()    │  │  sleep()    │  │  sleep()    │
└─────────────┘  │  bark()     │  │  meow()     │
                 └─────────────┘  └─────────────┘
```
---

**Method Call Implementation:**
```mips
# animal.speak()
lw $t0, 8($a0)      # Load vtable pointer from object
lw $t1, 0($t0)      # Load speak() method address
jalr $t1            # Jump to method
```

---

## **Modern Code Generation Trends**

### **Beyond Simple Stack Machines**

**Single Static Assignment (SSA):**
```llvm
; LLVM IR - Explicit data flow
%1 = add i32 %x, %y
%2 = mul i32 %1, 2
ret i32 %2
```

---

**Just-In-Time (JIT) Compilation:**
- **Profile-guided optimization**: Optimize based on runtime behavior
- **Deoptimization**: Fall back to interpreter for unexpected cases
- **Tiered compilation**: Start fast, optimize hot code paths

---

**WebAssembly:**
```wat
;; WebAssembly text format
(func $add (param $x i32) (param $y i32) (result i32)
  local.get $x
  local.get $y
  i32.add)
```

---

## **Project Implementation Patterns**

### **Code Generation for Different Paths**

**C/Flex/Bison Path:**
```c
void generate_code(ASTNode* node, FILE* output) {
    switch (node->type) {
        case NODE_NUMBER:
            fprintf(output, "  li $a0, %d\n", node->value);
            break;
        case NODE_ADD:
            generate_code(node->left, output);
            fprintf(output, "  sw $a0, 0($sp)\n");
            fprintf(output, "  addiu $sp, $sp, -4\n");
            generate_code(node->right, output);
            fprintf(output, "  lw $t1, 4($sp)\n");
            fprintf(output, "  add $a0, $t1, $a0\n");
            fprintf(output, "  addiu $sp, $sp, 4\n");
            break;
    }
}
```
---

**Python/PLY Path:**
```python
class CodeGenerator:
    def visit_BinaryOp(self, node):
        self.visit(node.left)
        self.emit("sw $a0, 0($sp)")
        self.emit("addiu $sp, $sp, -4")
        self.visit(node.right)
        self.emit("lw $t1, 4($sp)")
        self.emit(f"add $a0, $t1, $a0  # {node.op}")
        self.emit("addiu $sp, $sp, 4")
```
---

**OCaml/LLVM Path:**
```ocaml
let rec codegen expr =
  match expr with
  | Number n -> const_int i32_type n
  | Add (lhs, rhs) ->
      let l = codegen lhs in
      let r = codegen rhs in
      build_add l r "addtmp" builder
  | _ -> failwith "not implemented"
```

---

## **Optimization Techniques**

### **From Naive to Efficient Code Generation**

**Naive Stack Code:**
```mips
li $a0, 5
sw $a0, 0($sp)    # Unnecessary store/load
addiu $sp, $sp, -4
li $a0, 3
lw $t1, 4($sp)
add $a0, $t1, $a0
addiu $sp, $sp, 4
```
---

**Optimized Register Code:**
```mips
li $t0, 5         # Keep in register
li $a0, 3
add $a0, $t0, $a0 # Direct computation
```
---

**Common Optimizations:**
- **Register allocation**: Minimize memory traffic
- **Common subexpression elimination**: Reuse computed values
- **Constant folding**: Compute constants at compile time
- **Dead code elimination**: Remove unused computations

---

## **Real-World Code Generation Examples**

### **Industry Compiler Output**

**GCC for Simple C:**
```c
// C source
int calculate(int x, int y) {
    return x * y + 42;
}
```

```assembly
; GCC x86-64 output (optimized)
calculate:
    mov eax, edi
    imul eax, esi
    add eax, 42
    ret
```

---

**LLVM for the Same Function:**
```llvm
; LLVM IR
define i32 @calculate(i32 %x, i32 %y) {
  %1 = mul i32 %x, %y
  %2 = add i32 %1, 42
  ret i32 %2
}
```

---

## **Project Path Guidance - Code Generation**

*(see examples, 5-bison-example/with_codegen and 8-semantic analysis-type-checking)
You need to combine these two 
and replace stack operations with MIPS instructions*


### **Path: C/Flex/Bison - MIPS Assembly Generation**

**Recommended Approach:**
1. **Direct MIPS generation** from AST
2. **Use stack-based evaluation** initially
3. **Implement register allocation** as optimization

---

**Example Structure:**
```c
// codegen.h
typedef struct {
    FILE* output;
    int temp_counter;
    int label_counter;
} CodeGenContext;

void cgen_expression(CodeGenContext* ctx, ASTNode* node);
void cgen_statement(CodeGenContext* ctx, ASTNode* node);
void cgen_function(CodeGenContext* ctx, ASTNode* node);
```
---

```c
// Example implementation
void cgen_binary_op(CodeGenContext* ctx, ASTNode* node) {
    cgen_expression(ctx, node->left);
    fprintf(ctx->output, "  sw $a0, 0($sp)\n");
    fprintf(ctx->output, "  addiu $sp, $sp, -4\n");
    
    cgen_expression(ctx, node->right);
    
    fprintf(ctx->output, "  lw $t1, 4($sp)\n");
    
    switch(node->op) {
        case OP_ADD:
            fprintf(ctx->output, "  add $a0, $t1, $a0\n");
            break;
        case OP_SUB:
            fprintf(ctx->output, "  sub $a0, $t1, $a0\n");
            break;
        // ... other operators
    }
    
    fprintf(ctx->output, "  addiu $sp, $sp, 4\n");
}
```

---

**Milestones:**
1. Generate code for simple expressions
2. Add variable support (stack offsets)
3. Implement control flow (if/while)
4. Add function calls
5. Optimize (register allocation)

---

### **Path: Python/PLY - Interpreter with Bytecode**

**Recommended Approach:**
1. **Virtual machine** with bytecode
2. **Stack-based interpreter**
3. **Optionally generate MIPS** from bytecode

---

**Example Structure:**
```python
class BytecodeGenerator:
    def __init__(self):
        self.code = []
        self.constants = []
        self.labels = {}
    
    def visit_number(self, node):
        # Load constant
        const_index = self.add_constant(node.value)
        self.emit(('LOAD_CONST', const_index))
    
    def visit_binary_op(self, node):
        # Post-order: left, right, operator
        self.visit(node.left)
        self.visit(node.right)
        
        if node.op == '+':
            self.emit('ADD')
        elif node.op == '*':
            self.emit('MUL')
        # ... other operators
    
    def emit(self, instruction):
        self.code.append(instruction)
    
    def add_constant(self, value):
        self.constants.append(value)
        return len(self.constants) - 1
```

---

```python
# Virtual Machine
class VM:
    def __init__(self):
        self.stack = []
        self.constants = []
    
    def run(self, bytecode):
        for instr in bytecode.code:
            if instr[0] == 'LOAD_CONST':
                self.stack.append(self.constants[instr[1]])
            elif instr == 'ADD':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(a + b)
            # ... other instructions
```

---

**Milestones:**
1. Design bytecode instruction set
2. Implement bytecode generator from AST
3. Build stack-based virtual machine
4. Add control flow instructions
5. Add function call support

---

### **Path: OCaml/LLVM - LLVM IR Generation**

**Recommended Approach:**
1. **Generate LLVM Intermediate Representation**
2. **Let LLVM handle optimization and codegen**
3. **Focus on correct IR generation**
   

---

**Example Structure:**
```ocaml
(* LLVM Code Generation *)
module L = Llvm
module A = Ast

let the_module = L.create_module context "minilang"
let builder = L.builder context

let rec codegen_expr = function
  | A.Number n -> L.const_int i32_type n
  | A.Variable name ->
      (* Look up variable in symbol table *)
      let var = Symbol_table.find name !current_scope in
      L.build_load var name builder
  | A.BinaryOp (op, lhs, rhs) ->
      let l = codegen_expr lhs in
      let r = codegen_expr rhs in
      match op with
      | "+" -> L.build_add l r "addtmp" builder
      | "*" -> L.build_mul l r "multmp" builder
      | _ -> failwith "Unknown operator"
  
  | A.Call (func_name, args) ->
      let f = L.lookup_function func_name the_module in
      let args' = List.map codegen_expr args in
      L.build_call f (Array.of_list args') "calltmp" builder
```

---

```ocaml
(* Function generation *)
let codegen_function func =
  let param_types = Array.make (List.length func.params) i32_type in
  let ft = L.function_type i32_type param_types in
  let f = L.define_function func.name ft the_module in
  
  (* Set up builder and symbol table *)
  let builder = L.builder_at_end context (L.entry_block f) in
  
  (* Generate function body *)
  let body_val = codegen_expr func.body builder in
  ignore (L.build_ret body_val builder)
```

---

**Milestones:**
1. Set up LLVM context and module
2. Generate LLVM IR for expressions
3. Add variable support (alloca/store/load)
4. Implement control flow (br, phi nodes)
5. Add function definitions and calls

---

## **Testing Your Code Generator**

### **Testing Strategies for Each Path**

**Path (C/MIPS):**
```bash
# 1. Generate MIPS assembly
./minilangc program.ml > program.s

# 2. Assemble with MIPS simulator (like SPIM)
spim -f program.s

# 3. Or use cross-compiler
mips-linux-gnu-gcc -static program.s -o program
qemu-mips program
```

---

**Path (Python/Bytecode):**
```python
# Unit test for code generation
def test_codegen():
    ast = parse("3 + 5 * 2")
    generator = BytecodeGenerator()
    generator.visit(ast)
    
    # Expected bytecode sequence
    expected = [
        ('LOAD_CONST', 0),  # 3
        ('LOAD_CONST', 1),  # 5
        ('LOAD_CONST', 2),  # 2
        'MUL',
        'ADD'
    ]
    
    assert generator.code == expected
    print("Test passed!")
```

---

**Path (OCaml/LLVM):**
```ocaml
(* Test with LLVM's built-in JIT *)
let test_codegen () =
  let expr = A.BinaryOp("+", A.Number 3, A.Number 5) in
  let ir_value = codegen_expr expr in
  
  (* Create a test function *)
  let ft = L.function_type i32_type [||] in
  let test_func = L.define_function "test" ft the_module in
  let builder = L.builder_at_end context (L.entry_block test_func) in
  ignore (L.build_ret ir_value builder);
  
  (* JIT compile and run *)
  let engine = L.ExecutionEngine.create the_module in
  let result = L.ExecutionEngine.run_function test_func [||] engine in
  let int_result = L.Int64.to_int (L.ExecutionEngine.GenericValue.as_int result) in
  
  assert (int_result = 8);
  print_endline "Test passed!"
```

---

## **Common Pitfalls & Solutions**

### **Code Generation Challenges**

| Problem | Solution | Example |
|---------|----------|---------|
| **Stack imbalance** | Count pushes/pops | Add `stack_depth` tracking |
| **Register spilling** | Save/restore temps | Use `$s0-$s7` for long-lived values |
| **Label generation** | Unique label names | Use counter: `label_1`, `label_2` |
| **Function prologue/epilogue** | Standard template | Always save `$fp`, `$ra` |
| **Type mismatches** | Runtime checks or static typing | Add type annotations in AST |

### **Debugging Generated Code:**
1. **Add comments** to generated code:
   ```mips
   # Generated for: x = y + 5
   lw $a0, 8($fp)      # Load y
   addi $a0, $a0, 5    # y + 5
   sw $a0, 4($fp)      # Store to x
   ```

2. **Step-by-step simulation**:
   ```python
   def simulate_mips(code):
       registers = {'$a0': 0, '$t0': 0, ...}
       memory = {}
       stack = []
       # Execute each instruction, print state
   ```

3. **Compare with reference**:
   - Use GCC/Clang to compile equivalent C code
   - Compare your MIPS output with compiler's output
   - Learn optimization techniques

---


## **Testing Your Code Generator**

### **Verification Strategies**

**Golden Testing:**
```python
def test_code_generation():
    source = "3 + 5 * 2"
    expected_asm = """
        li $a0, 3
        sw $a0, 0($sp)
        addiu $sp, $sp, -4
        li $a0, 5
        sw $a0, 0($sp)
        addiu $sp, $sp, -4
        li $a0, 2
        lw $t1, 4($sp)
        mul $a0, $t1, $a0
        addiu $sp, $sp, 4
        lw $t1, 4($sp)
        add $a0, $t1, $a0
        addiu $sp, $sp, 4
    """
    assert generate_code(parse(source)) == normalize_asm(expected_asm)
```
---

**Execution Testing:**
```python
def test_execution():
    program = "def main() = 3 + 5;"
    asm = generate_code(parse(program))
    result = run_on_simulator(asm)
    assert result == 8
```

---

## **Key Takeaways**

### **Code Generation Principles**

1. **Abstraction Layers**: AST → Intermediate Representation → Target Code
2. **Pattern Matching**: Each AST node type has a code generation template
3. **Resource Management**: Stack vs registers, memory layout decisions
4. **Runtime Integration**: Code must work with your runtime environment design
5. **Testing**: Generated code must be verified for correctness

---

### **Modern Relevance:**
- **WebAssembly**: New target for web compilation
- **JIT Compilation**: Runtime code generation and optimization
- **Cross-compilation**: Generating code for different architectures
- **Formal Verification**: Proving correctness of generated code

---

## **Next Steps & Resources**

### **Continuing Compiler Journey**

**Advanced Topics (next weeks):**
- **Register Allocation** (graph coloring, linear scan)
- **Instruction Selection** (tree pattern matching)
- **Peephole Optimization** (local pattern-based improvements)
- **Vectorization** (SIMD instruction generation)

**Tools & Libraries:**
- **LLVM**: Industrial-strength code generation framework
- **GNU Lightning**: Lightweight JIT compilation library
- **Cranelift**: Modern code generator written in Rust
- **DynASM**: Dynamic assembler for JIT compilers

---

**Further Reading:**
- "Engineering a Compiler" by Cooper & Torczon
- "Advanced Compiler Design and Implementation" by Muchnick
- LLVM Documentation: https://llvm.org/docs/

