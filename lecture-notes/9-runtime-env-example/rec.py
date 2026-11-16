import sys
def depth(n): return depth(n+1)
def fact(n): return 1 if n <= 1 else n * fact(n-1)
print(sys.getrecursionlimit())
try: fact(5000)
except RecursionError as e: print("RecursionError:", e)
