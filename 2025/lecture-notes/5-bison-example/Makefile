CC = gcc
CFLAGS = -g

all: calculator

calculator: lex.yy.c parser.tab.c
	$(CC) $(CFLAGS) -o $@ $^

lex.yy.c: lexer.l parser.tab.h
	flex $<

parser.tab.c parser.tab.h: parser.y
	bison -d $<

clean:
	rm -f calculator lex.yy.c parser.tab.* *.o

test: calculator
	./calculator


# Step 1: Generate parser files
# bison -d parser.y        # Creates parser.tab.c and parser.tab.h

# Step 2: Generate lexer
# flex lexer.l             # Creates lex.yy.c

# Step 3: Compile everything
# gcc -o calculator parser.tab.c lex.yy.c -lm
# -lm links the math library if needed