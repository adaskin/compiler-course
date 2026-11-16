#!/usr/bin/env python3
"""Tiny MIPS-subset simulator used only to test mips_cgen.py (real tools: SPIM, MARS)."""
import re, sys
def wrap(n): return (n + 2**31) % 2**32 - 2**31
def run(asm, max_steps=5_000_000):
    prog, labels = [], {}
    for line in asm.splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("."): continue
        if line.endswith(":"): labels[line[:-1]] = len(prog); continue
        op, _, rest = line.partition(" ")
        prog.append((op, [a.strip() for a in rest.split(",")] if rest else []))
    R = {r: 0 for r in ["$zero", "$a0", "$t1", "$v0", "$sp", "$fp", "$ra", "lo"]}
    R["$sp"] = 0x7ffffffc; mem, out = {}, []
    def val(a): return R[a] if a.startswith("$") else int(a)
    def addr(a):
        m = re.match(r"(-?\d+)\((\$\w+)\)", a); return R[m.group(2)] + int(m.group(1))
    pc = labels["main"]; steps = 0
    while True:
        steps += 1; assert steps < max_steps, "too many steps"
        op, a = prog[pc]; pc += 1
        if op == "li": R[a[0]] = int(a[1])
        elif op == "move": R[a[0]] = R[a[1]]
        elif op == "lw": R[a[0]] = mem[addr(a[1])]
        elif op == "sw": mem[addr(a[1])] = R[a[0]]
        elif op == "addiu": R[a[0]] = wrap(R[a[1]] + int(a[2]))
        elif op == "add": R[a[0]] = wrap(R[a[1]] + R[a[2]])
        elif op == "sub": R[a[0]] = wrap(R[a[1]] - R[a[2]])
        elif op == "mul": R[a[0]] = wrap(R[a[1]] * R[a[2]])
        elif op == "div": b = R[a[1]]; x = R[a[0]]; q = abs(x) // abs(b); R["lo"] = q if (x < 0) == (b < 0) else -q
        elif op == "mflo": R[a[0]] = R["lo"]
        elif op == "slt": R[a[0]] = int(R[a[1]] < R[a[2]])
        elif op == "sltu": R[a[0]] = int((R[a[1]] & 0xffffffff) < (R[a[2]] & 0xffffffff))
        elif op == "sltiu": R[a[0]] = int((R[a[1]] & 0xffffffff) < int(a[2]))
        elif op == "xor": R[a[0]] = R[a[1]] ^ R[a[2]]
        elif op == "xori": R[a[0]] = R[a[1]] ^ int(a[2])
        elif op == "beq": pc = labels[a[2]] if R[a[0]] == R[a[1] if a[1].startswith("$") else "$zero"] else pc
        elif op == "bne": pc = labels[a[2]] if R[a[0]] != R[a[1]] else pc
        elif op == "b": pc = labels[a[0]]
        elif op == "jal": R["$ra"] = pc; pc = labels[a[0]]
        elif op == "jr": pc = R[a[0]]
        elif op == "syscall":
            v = R["$v0"]
            if v == 1: out.append(str(R["$a0"]))
            elif v == 11: out.append(chr(R["$a0"]))
            elif v == 10: return "".join(out), 0
            elif v == 17: return "".join(out), R["$a0"]
        else: raise NotImplementedError(op)
if __name__ == "__main__":
    o, rc = run(open(sys.argv[1]).read()); sys.stdout.write(o); sys.exit(rc)
