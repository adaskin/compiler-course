#!/usr/bin/env python3
"""A very small compiler for the course's mini functional language.
Pipeline: tokens -> AST -> typed AST -> textual LLVM IR (opaque pointers, LLVM 15+).
Meant to be read next to lecture 10, not to be a reference solution:
no error recovery, no diagnostics beyond a Python exception."""
import re, sys

# ---------- 1. lexer ---------------------------------------------------
KEYWORDS = {"let", "in", "if", "then", "else", "fun", "true", "false", "int", "bool"}
TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_]\w*)|(->|<=|>=|==|!=|&&|\|\||[-+*/<>():=]))")

def lex(src):
    toks, pos, line_start = [], 0, 0
    while pos < len(src):
        m = TOKEN.match(src, pos)
        if not m:
            if src[pos:].strip() == "": break
            raise SyntaxError(f"bad character at {pos}")
        kind = "int" if m.group(1) else "id" if m.group(2) else "op"
        text = m.group(0).strip()
        start = m.end() - len(text)
        col = start - (src.rfind("\n", 0, start) + 1)
        toks.append((kind, text, col))
        pos = m.end()
    toks.append(("eof", "", 0))
    return toks

# ---------- 2. parser (recursive descent, AST = tuples) -------------------
class Parser:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i][1] if self.t[self.i][0] != "int" else "#int"
    def kind(self): return self.t[self.i][0]
    def next(self): tok = self.t[self.i]; self.i += 1; return tok[1]
    def eat(self, s):
        if self.t[self.i][1] != s: raise SyntaxError(f"expected {s!r}, found {self.t[self.i][1]!r}")
        return self.next()

    def program(self):
        defs = []
        while self.peek() == "let":
            save = self.i
            d = self.definition()
            if self.peek() == "in":          # it was the start of the final expression
                self.i = save; break
            defs.append(d)
        return defs, self.expr()

    def definition(self):
        self.eat("let"); name = self.next(); params = []
        while self.peek() == "(":
            self.next(); p = self.next(); self.eat(":"); params.append((p, self.type())); self.eat(")")
        ret = None
        if self.peek() == ":": self.next(); ret = self.type()
        self.eat("="); return ("def", name, params, ret, self.expr())

    def type(self):
        a = self.atype()
        if self.peek() == "->":
            self.next(); r = self.type()
            return ("fn", [a] + (r[1] if r[0] == "fn" else []), r[2] if r[0] == "fn" else r)
        return a
    def atype(self):
        if self.peek() == "(":
            self.next(); t = self.type(); self.eat(")"); return t
        return self.next()                   # "int" | "bool"

    def expr(self):
        if self.peek() == "let":
            self.next(); x = self.next(); self.eat("="); e1 = self.expr(); self.eat("in")
            return ("let", x, e1, self.expr())
        if self.peek() == "if":
            self.next(); c = self.expr(); self.eat("then"); a = self.expr(); self.eat("else")
            return ("if", c, a, self.expr())
        if self.peek() == "fun":
            self.next(); self.eat("("); x = self.next(); self.eat(":"); t = self.type(); self.eat(")")
            self.eat("->"); return ("fun", x, t, self.expr())
        return self.binary(0)

    LEVELS = [["||"], ["&&"], ["<", "<=", ">", ">=", "==", "!="], ["+", "-"], ["*", "/"]]
    def binary(self, lvl):
        if lvl == len(self.LEVELS): return self.unary()
        left = self.binary(lvl + 1)
        while self.peek() in self.LEVELS[lvl]:
            op = self.next(); left = ("bin", op, left, self.binary(lvl + 1))
            if lvl == 2: break               # comparisons do not chain
        return left
    def unary(self):
        if self.peek() == "-": self.next(); return ("neg", self.unary())
        return self.app()
    def app(self):
        f = self.atom(); args = []
        # layout rule: a token at column 0 starts the next top-level item
        while self.starts_atom() and self.t[self.i][2] != 0: args.append(self.atom())
        return ("app", f, args) if args else f
    def starts_atom(self):
        return self.kind() in ("int", "id") and self.peek() not in KEYWORDS - {"true", "false"} \
               or self.peek() in ("(", "true", "false")
    def atom(self):
        if self.kind() == "int": return ("int", int(self.next()))
        if self.peek() in ("true", "false"): return ("bool", self.next() == "true")
        if self.peek() == "(":
            self.next(); e = self.expr(); self.eat(")"); return e
        return ("var", self.next())

# ---------- 3. type checker (week 8): returns a *typed* AST ---------------
BUILTINS = {"print_int": ("fn", ["int"], "int"), "print_bool": ("fn", ["bool"], "int")}

def check(e, env):
    k = e[0]
    if k == "int":  return e, "int"
    if k == "bool": return e, "bool"
    if k == "var":
        if e[1] not in env: raise TypeError(f"unbound variable {e[1]}")
        return ("var", e[1], env[e[1]]), env[e[1]]
    if k == "neg":
        a, t = check(e[1], env); assert t == "int"; return ("neg", a), "int"
    if k == "bin":
        op = e[1]; a, ta = check(e[2], env); b, tb = check(e[3], env)
        assert ta == tb, f"operands of {op}: {ta} vs {tb}"
        if op in ("+", "-", "*", "/"):  assert ta == "int"; return ("bin", op, a, b, ta), "int"
        if op in ("<", "<=", ">", ">="): assert ta == "int"; return ("bin", op, a, b, ta), "bool"
        if op in ("==", "!="): assert ta in ("int", "bool"); return ("bin", op, a, b, ta), "bool"
        assert ta == "bool"; return ("bin", op, a, b, ta), "bool"         # && ||
    if k == "if":
        c, tc = check(e[1], env); a, ta = check(e[2], env); b, tb = check(e[3], env)
        assert tc == "bool" and ta == tb, "if: condition must be bool, branches must agree"
        return ("if", c, a, b, ta), ta
    if k == "let":
        a, ta = check(e[2], env); b, tb = check(e[3], {**env, e[1]: ta})
        return ("let", e[1], a, b), tb
    if k == "fun":
        body, tb = check(e[3], {**{n: t for n, t in env.items() if n in GLOBALS}, e[1]: e[2]})
        return ("fun", e[1], e[2], body), ("fn", [e[2]], tb)      # no capture: only globals visible
    if k == "app":
        f, tf = check(e[1], env); assert tf[0] == "fn" and len(tf[1]) == len(e[2]), "bad call"
        args = []
        for a, pt in zip(e[2], tf[1]):
            ca, ta = check(a, env); assert ta == pt, f"argument: expected {pt}, found {ta}"; args.append(ca)
        return ("app", f, args, tf), tf[2]
    raise TypeError(k)

GLOBALS = set()

def check_program(defs, final):
    env = dict(BUILTINS); GLOBALS.update(env)
    for _, name, params, ret, _ in defs:                      # pass 1: signatures
        if params: env[name] = ("fn", [t for _, t in params], ret); GLOBALS.add(name)
    tdefs = []
    for _, name, params, ret, body in defs:                   # pass 2: bodies
        local = {**env, **{p: t for p, t in params}}
        tb, t = check(body, local)
        if params: assert t == ret, f"{name}: body has type {t}, declared {ret}"
        else: env[name] = t                                   # top-level value, visible to later items
        tdefs.append((name, params, ret or t, tb))
    tf, _ = check(final, env)
    return tdefs, tf

# ---------- 4. code generator: typed AST -> textual LLVM IR --------------
LL = {"int": "i32", "bool": "i1"}
def llty(t): return "ptr" if isinstance(t, tuple) else LL[t]

class Fn:
    """State for one LLVM function being emitted."""
    def __init__(self, mod): self.mod, self.out, self.n, self.cur = mod, [], 0, "entry"
    def tmp(self): self.n += 1; return f"%t.{self.n}"
    def lab(self, base): self.n += 1; return f"{base}.{self.n}"
    def emit(self, s): self.out.append("  " + s)
    def start(self, label): self.out.append(f"{label}:"); self.cur = label
    def br(self, label): self.emit(f"br label %{label}")

    def gen(self, e, env):                    # returns an LLVM operand (a constant or %name)
        k = e[0]
        if k == "int":  return str(e[1])
        if k == "bool": return "true" if e[1] else "false"
        if k == "var":
            if e[1] in env: return env[e[1]]  # let-bound names are just SSA values: no memory
            return f"@{e[1]}"                 # a top-level function used as a value: a ptr
        if k == "neg":
            t = self.tmp(); self.emit(f"{t} = sub i32 0, {self.gen(e[1], env)}"); return t
        if k == "let":
            v = self.gen(e[2], env); return self.gen(e[3], {**env, e[1]: v})
        if k == "fun":                        # lambda lifting: the lambda becomes a top-level function
            return self.mod.lift(e)
        if k == "if":                         # condition, two branches, phi
            c = self.gen(e[1], env); th, el, jn = self.lab("then"), self.lab("else"), self.lab("join")
            self.emit(f"br i1 {c}, label %{th}, label %{el}")
            self.start(th); a = self.gen(e[2], env); th_end = self.cur; self.br(jn)
            self.start(el); b = self.gen(e[3], env); el_end = self.cur; self.br(jn)
            self.start(jn); t = self.tmp()    # th_end / el_end, NOT th / el: branches may contain ifs
            self.emit(f"{t} = phi {llty(e[4])} [ {a}, %{th_end} ], [ {b}, %{el_end} ]"); return t
        if k == "app":
            return self.call(e, env)
        if k == "bin":
            return self.binary(e, env)
        raise NotImplementedError(k)

    def binary(self, e, env):
        _, op, l, r, ot = e
        if op in ("&&", "||"):                # short-circuit: the right operand may not run
            a = self.gen(l, env); lhs_end = self.cur
            rhs, jn = self.lab("rhs"), self.lab("join")
            if op == "&&": self.emit(f"br i1 {a}, label %{rhs}, label %{jn}"); short = "false"
            else:          self.emit(f"br i1 {a}, label %{jn}, label %{rhs}"); short = "true"
            self.start(rhs); b = self.gen(r, env); rhs_end = self.cur; self.br(jn)
            self.start(jn); t = self.tmp()
            self.emit(f"{t} = phi i1 [ {short}, %{lhs_end} ], [ {b}, %{rhs_end} ]"); return t
        a, b = self.gen(l, env), self.gen(r, env); t = self.tmp()
        if op in ("+", "-", "*"):             # i32 arithmetic wraps: no nsw flag
            self.emit(f"{t} = {dict(zip('+-*', ['add', 'sub', 'mul']))[op]} i32 {a}, {b}")
        elif op == "/":                       # sdiv by zero is undefined: check first
            z, bad, ok = self.tmp(), self.lab("divzero"), self.lab("divok")
            self.emit(f"{z} = icmp eq i32 {b}, 0"); self.emit(f"br i1 {z}, label %{bad}, label %{ok}")
            self.start(bad); self.emit("call void @fpl_div_zero()"); self.emit("unreachable")
            self.start(ok); self.emit(f"{t} = sdiv i32 {a}, {b}")
        else:
            pred = {"<": "slt", "<=": "sle", ">": "sgt", ">=": "sge", "==": "eq", "!=": "ne"}[op]
            self.emit(f"{t} = icmp {pred} {llty(ot)} {a}, {b}")
        return t

    def call(self, e, env):
        _, f, args, ft = e
        vals = [self.gen(a, env) for a in args]
        if f[0] == "var" and f[1] in ("print_int", "print_bool"):
            if f[1] == "print_bool":                       # the C runtime takes an int
                z = self.tmp(); self.emit(f"{z} = zext i1 {vals[0]} to i32"); vals = [z]
            t = self.tmp(); self.emit(f"{t} = call i32 @{f[1]}(i32 {vals[0]})"); return t
        callee = f"@{f[1]}" if f[0] == "var" and f[1] in self.mod.globals else self.gen(f, env)
        argl = ", ".join(f"{llty(t)} {v}" for t, v in zip(ft[1], vals))
        t = self.tmp(); self.emit(f"{t} = call {llty(ft[2])} {callee}({argl})"); return t

class Module:
    def __init__(self): self.funcs, self.globals, self.lams = [], set(), 0

    def function(self, name, params, ret, body, env):
        f = Fn(self); f.start("entry")
        penv = {**env, **{p: f"%{p}" for p, _ in params}}
        v = f.gen(body, penv); f.emit(f"ret {llty(ret)} {v}")
        sig = ", ".join(f"{llty(t)} %{p}" for p, t in params)
        self.funcs.append(f"define {llty(ret)} @{name}({sig}) {{\n" + "\n".join(f.out) + "\n}")

    def lift(self, e):                        # ("fun", x, ty, body): body type is recomputed from codegen
        self.lams += 1; name = f"lambda.{self.lams}"
        self.function(name, [(e[1], e[2])], self.result_type(e[3]), e[3], self.genv)
        return f"@{name}"
    def result_type(self, body):
        k = body[0]
        if k in ("int", "neg"): return "int"
        if k == "bool": return "bool"
        if k == "var": return body[2]
        if k in ("if",): return body[4]
        if k == "let": return self.result_type(body[3])
        if k == "app": return body[3][2]
        if k == "bin": return "bool" if body[1] in ("<", "<=", ">", ">=", "==", "!=", "&&", "||") else "int"
        raise NotImplementedError(k)

def compile_program(src):
    defs, final = Parser(lex(src)).program()
    tdefs, tfinal = check_program(defs, final)
    m = Module(); m.globals = {n for n, ps, _, _ in tdefs if ps} | {"print_int", "print_bool"}
    m.genv = {}                               # lambdas see only top-level functions (no capture)
    for name, params, ret, body in tdefs:
        if params: m.function(name, params, ret, body, {})
    main = Fn(m); main.start("entry"); env = {}
    for name, params, ret, body in tdefs:     # top-level values, in order, then the final expression
        if not params: env[name] = main.gen(body, env)
    main.gen(tfinal, env); main.emit("ret i32 0")
    return ("declare i32 @print_int(i32)\ndeclare i32 @print_bool(i32)\ndeclare void @fpl_div_zero()\n\n"
            + "\n\n".join(m.funcs) + "\n\ndefine i32 @main() {\n" + "\n".join(main.out) + "\n}\n")

if __name__ == "__main__":
    print(compile_program(open(sys.argv[1]).read()), end="")
