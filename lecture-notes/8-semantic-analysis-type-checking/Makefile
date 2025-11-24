CC = gcc
CFLAGS = -g -Wall
FLEX = flex
BISON = bison

all: compiler

compiler: lex.yy.o parser.tab.o ast.o symbol.o type.o main.o
	$(CC) $(CFLAGS) -o $@ $^

lex.yy.o: lex.yy.c parser.tab.h
	$(CC) $(CFLAGS) -c $<

parser.tab.o: parser.tab.c
	$(CC) $(CFLAGS) -c $<

ast.o: ast.c ast.h
	$(CC) $(CFLAGS) -c ast.c

symbol.o: symbol.c symbol.h type.h
	$(CC) $(CFLAGS) -c symbol.c

type.o: type.c type.h
	$(CC) $(CFLAGS) -c type.c

main.o: main.c ast.h symbol.h type.h
	$(CC) $(CFLAGS) -c main.c

lex.yy.c: lexer.l
	$(FLEX) $<

parser.tab.c parser.tab.h: parser.y
	$(BISON) -d $<

clean:
	rm -f compiler lex.yy.c parser.tab.c parser.tab.h *.o

test: compiler
	@echo "int x; x = 5 + 3; x = x * 2;" > test.txt
	./compiler test.txt

.PHONY: all clean test

# # Generate parser
# bison -d parser.y

# # Generate lexer  
# flex lexer.l

# # Compile everything
# gcc -c ast.c -o ast.o
# gcc -c type.c -o type.o
# gcc -c symbol.c -o symbol.o
# gcc -c main.c -o main.o
# gcc -c parser.tab.c -o parser.tab.o
# gcc -c lex.yy.c -o lex.yy.o

# # Link
# gcc ast.o type.o symbol.o main.o parser.tab.o lex.yy.o -o compiler

# # Run with test file
# ./compiler test_input.txt