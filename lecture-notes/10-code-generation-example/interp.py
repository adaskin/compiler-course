#!/usr/bin/env python3
"""Reference interpreter over the typed AST: the oracle for differential testing."""
import sys
from fplc_mini import Parser, lex, check_program

class DivZero(Exception): pass

def wrap(n): return (n + 2**31) % 2**32 - 2**31

def ev(e, env, glob, out):
    k = e[0]
    if k in ("int", "bool"): return e[1]
    if k == "var": return env[e[1]] if e[1] in env else glob[e[1]]
    if k == "neg": return wrap(-ev(e[1], env, glob, out))
    if k == "let": return ev(e[3], {**env, e[1]: ev(e[2], env, glob, out)}, glob, out)
    if k == "if": return ev(e[2] if ev(e[1], env, glob, out) else e[3], env, glob, out)
    if k == "fun": return ("closure", e[1], e[3])
    if k == "bin":
        op = e[1]
        if op == "&&": return ev(e[2], env, glob, out) and ev(e[3], env, glob, out)
        if op == "||": return ev(e[2], env, glob, out) or ev(e[3], env, glob, out)
        a, b = ev(e[2], env, glob, out), ev(e[3], env, glob, out)
        if op == "/":
            if b == 0: raise DivZero
            q = abs(a) // abs(b); return wrap(q if (a < 0) == (b < 0) else -q)   # truncates toward zero
        return {"+": lambda: wrap(a + b), "-": lambda: wrap(a - b), "*": lambda: wrap(a * b),
                "<": lambda: a < b, "<=": lambda: a <= b, ">": lambda: a > b, ">=": lambda: a >= b,
                "==": lambda: a == b, "!=": lambda: a != b}[op]()
    if k == "app":
        f = e[1]
        if f[0] == "var" and f[1] in ("print_int", "print_bool"):
            v = ev(e[2][0], env, glob, out); out.append(str(v).lower() if f[1] == "print_bool" else str(v)); return 0
        fv = ev(f, env, glob, out); args = [ev(a, env, glob, out) for a in e[2]]
        if fv[0] == "closure": return ev(fv[2], {**glob, fv[1]: args[0]}, glob, out)
        _, params, body = fv
        return ev(body, {**glob, **dict(zip([p for p, _ in params], args))}, glob, out)
    raise NotImplementedError(k)

def run(src):
    defs, final = Parser(lex(src)).program()
    tdefs, tfinal = check_program(defs, final)
    glob, out = {}, []
    for name, params, ret, body in tdefs:
        if params: glob[name] = ("fn", params, body)
    try:
        env = {}
        for name, params, ret, body in tdefs:
            if not params: env[name] = ev(body, env, glob, out)
        ev(tfinal, env, glob, out)
        return "".join(l + "\n" for l in out), 0
    except DivZero:
        return "".join(l + "\n" for l in out), 1

if __name__ == "__main__":
    o, rc = run(open(sys.argv[1]).read()); sys.stdout.write(o); sys.exit(rc)
