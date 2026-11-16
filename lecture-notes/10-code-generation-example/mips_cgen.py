#!/usr/bin/env python3
"""Stack-machine code generator (accumulator in $a0, temporaries on the stack) for the
subset int/bool, let, if, && ||, arithmetic, comparison, calls to top-level functions.
Convention (Cool / Stanford CS143): caller pushes $fp, then the arguments last-to-first;
the callee points $fp at its return address and pops its own arguments."""
import sys
from fplc_mini import Parser, lex, check_program

class MipsGen:
    def __init__(self): self.out, self.n = [], 0
    def emit(self, s): self.out.append(s if s.endswith(":") else "    " + s)
    def label(self, base): self.n += 1; return f"{base}_{self.n}"
    def push(self): self.emit("sw   $a0, 0($sp)"); self.emit("addiu $sp, $sp, -4")
    def pop_to(self, reg): self.emit(f"lw   {reg}, 4($sp)"); self.emit("addiu $sp, $sp, 4")

    # env maps a name to its offset from $fp; d = words pushed since the body started
    def cgen(self, e, env, d):
        k = e[0]
        if k == "int":  self.emit(f"li   $a0, {e[1]}")
        elif k == "bool": self.emit(f"li   $a0, {1 if e[1] else 0}")
        elif k == "var": self.emit(f"lw   $a0, {env[e[1]]}($fp)")
        elif k == "neg": self.cgen(e[1], env, d); self.emit("sub  $a0, $zero, $a0")
        elif k == "let":
            self.cgen(e[2], env, d); self.push()                 # x lives in the slot just pushed
            self.cgen(e[3], {**env, e[1]: -4 * (d + 1)}, d + 1)
            self.emit("addiu $sp, $sp, 4")                       # drop x; the result stays in $a0
        elif k == "if":
            els, end = self.label("else"), self.label("endif")
            self.cgen(e[1], env, d); self.emit(f"beq  $a0, $zero, {els}")
            self.cgen(e[2], env, d); self.emit(f"b    {end}")
            self.emit(f"{els}:"); self.cgen(e[3], env, d); self.emit(f"{end}:")
        elif k == "bin": self.binary(e, env, d)
        elif k == "app": self.call(e, env, d)
        else: raise NotImplementedError(k)

    def binary(self, e, env, d):
        op = e[1]
        if op in ("&&", "||"):                       # short-circuit: skip the right operand
            end = self.label("sc"); self.cgen(e[2], env, d)
            self.emit(f"{'beq' if op == '&&' else 'bne'}  $a0, $zero, {end}")
            self.cgen(e[3], env, d); self.emit(f"{end}:"); return
        self.cgen(e[2], env, d); self.push()         # e1 -> stack
        self.cgen(e[3], env, d + 1); self.pop_to("$t1")   # e2 in $a0, e1 in $t1
        if op == "+":  self.emit("add  $a0, $t1, $a0")
        elif op == "-": self.emit("sub  $a0, $t1, $a0")
        elif op == "*": self.emit("mul  $a0, $t1, $a0")
        elif op == "/":
            self.emit(f"beq  $a0, $zero, div_by_zero"); self.emit("div  $t1, $a0"); self.emit("mflo $a0")
        elif op == "<":  self.emit("slt  $a0, $t1, $a0")
        elif op == ">":  self.emit("slt  $a0, $a0, $t1")
        elif op == "<=": self.emit("slt  $a0, $a0, $t1"); self.emit("xori $a0, $a0, 1")
        elif op == ">=": self.emit("slt  $a0, $t1, $a0"); self.emit("xori $a0, $a0, 1")
        elif op == "==": self.emit("xor  $a0, $t1, $a0"); self.emit("sltiu $a0, $a0, 1")
        elif op == "!=": self.emit("xor  $a0, $t1, $a0"); self.emit("sltu $a0, $zero, $a0")

    def call(self, e, env, d):
        f, args = e[1][1], e[2]
        if f in ("print_int", "print_bool"):
            self.cgen(args[0], env, d)
            self.emit("li   $v0, 1"); self.emit("syscall")           # SPIM: print_int($a0)
            self.emit("li   $a0, 10"); self.emit("li   $v0, 11"); self.emit("syscall")   # newline
            self.emit("li   $a0, 0"); return                         # the built-ins return 0
        self.emit("sw   $fp, 0($sp)"); self.emit("addiu $sp, $sp, -4")        # save caller's $fp
        for i, a in enumerate(reversed(args)):                                 # last argument first
            self.cgen(a, env, d + 1 + i); self.push()
        self.emit(f"jal  {f}")                                                  # callee pops args and $fp

    def function(self, name, params, body):
        env = {p: 4 * (i + 1) for i, (p, _) in enumerate(params)}              # x at 4($fp), y at 8($fp), ...
        self.emit(f"{name}:")
        self.emit("move $fp, $sp"); self.emit("sw   $ra, 0($sp)"); self.emit("addiu $sp, $sp, -4")
        self.cgen(body, env, 1 - 1)
        self.emit("lw   $ra, 4($sp)"); self.emit(f"addiu $sp, $sp, {4 * len(params) + 8}")
        self.emit("lw   $fp, 0($sp)"); self.emit("jr   $ra")

def compile_mips(src):
    defs, final = Parser(lex(src)).program()
    tdefs, tfinal = check_program(defs, final)
    g = MipsGen()
    g.emit(".text"); g.emit(".globl main"); g.emit("main:")
    g.emit("move $fp, $sp"); g.emit("addiu $sp, $sp, -4")    # same frame shape as a function: slot 0($fp) is reserved
    env, d = {}, 0
    for name, params, ret, body in tdefs:                 # top-level values: one stack slot each
        if not params:
            g.cgen(body, env, d); g.push(); d += 1; env[name] = -4 * d
    g.cgen(tfinal, env, d)
    g.emit("li   $v0, 10"); g.emit("syscall")             # exit
    g.emit("div_by_zero:"); g.emit("li   $v0, 17"); g.emit("li   $a0, 1"); g.emit("syscall")   # exit2(1)
    for name, params, ret, body in tdefs:
        if params: g.function(name, params, body)
    return "\n".join(g.out) + "\n"

if __name__ == "__main__":
    print(compile_mips(open(sys.argv[1]).read()), end="")
