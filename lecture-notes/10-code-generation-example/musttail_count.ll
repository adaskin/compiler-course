declare i32 @print_int(i32)

define i32 @count(i32 %n, i32 %acc) {
entry:
  %c = icmp eq i32 %n, 0
  br i1 %c, label %done, label %rec
done:
  ret i32 %acc
rec:
  %n1 = sub i32 %n, 1
  %a1 = add i32 %acc, 1
  %r = musttail call i32 @count(i32 %n1, i32 %a1)
  ret i32 %r
}

define i32 @main() {
entry:
  %v = call i32 @count(i32 100000000, i32 0)
  %p = call i32 @print_int(i32 %v)
  ret i32 0
}
