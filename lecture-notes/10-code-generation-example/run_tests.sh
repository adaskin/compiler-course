#!/bin/bash
# Build and run every tests/*.fpl three ways and compare:
#   1. compiled with clang (no optimisation)        vs  hand-written .expected
#   2. the same program compiled with -O2           vs  .expected   (optimiser must not change behaviour)
#   3. the reference interpreter (interp.py)         vs  .expected
# Every emitted .ll must also pass the LLVM verifier.
cd "$(dirname "$0")"; mkdir -p build; pass=0; fail=0
for src in tests/*.fpl; do
  name=$(basename "$src" .fpl); exp="tests/$name.expected"; want_rc=0
  [ -f "tests/$name.exit" ] && want_rc=$(cat "tests/$name.exit")
  python3 fplc_mini.py "$src" > "build/$name.ll" || { echo "FAIL $name (compiler crashed)"; fail=$((fail+1)); continue; }
  opt -passes=verify -disable-output "build/$name.ll" || { echo "FAIL $name (verifier)"; fail=$((fail+1)); continue; }
  ok=1
  for opt_level in -O0 -O2; do
    clang -w $opt_level "build/$name.ll" runtime.c -o "build/$name$opt_level" 2>/dev/null
    got=$("build/$name$opt_level" 2>/dev/null); rc=$?
    [ "$got" == "$(cat $exp)" ] && [ "$rc" == "$want_rc" ] || { echo "FAIL $name ($opt_level): rc=$rc got: $got"; ok=0; }
  done
  got=$(python3 interp.py "$src" 2>/dev/null); rc=$?
  [ "$got" == "$(cat $exp)" ] && [ "$rc" == "$want_rc" ] || { echo "FAIL $name (interpreter): rc=$rc got: $got"; ok=0; }
  if [ $ok = 1 ]; then echo "PASS $name"; pass=$((pass+1)); else fail=$((fail+1)); fi
done
echo "$pass passed, $fail failed"; [ $fail = 0 ]
