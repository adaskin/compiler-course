CC = gcc
CFLAGS = -g -Wall

all: calculator

calculator: parser.tab.o lex.yy.o ast.o codegen.o
	$(CC) $(CFLAGS) -o calculator parser.tab.o lex.yy.o ast.o codegen.o -lm

ast.o: ast.c ast.h
	$(CC) $(CFLAGS) -c ast.c

codegen.o: codegen.c codegen.h ast.h
	$(CC) $(CFLAGS) -c codegen.c

parser.tab.o: parser.tab.c
	$(CC) $(CFLAGS) -c parser.tab.c

lex.yy.o: lex.yy.c
	$(CC) $(CFLAGS) -c lex.yy.c

parser.tab.c parser.tab.h: parser.y
	bison -d parser.y

lex.yy.c: lexer.l parser.tab.h
	flex lexer.l

clean:
	rm -f calculator *.o lex.yy.c parser.tab.*

test: calculator
	echo "2 + 3 * 4" | ./calculator

.PHONY: all clean test