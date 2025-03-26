CC = g++

CFLAGS = -g -O2 -I../../include -fopenmp
CFLAGS+= $(extra)
CFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align
CFLAGS+= -Wwrite-strings -Wmissing-declarations -Wredundant-decls
CFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized
CFLAGS+= -Wpedantic -Wno-unused-parameter # does not make much sense with function overloading

.PHONY: all clean

all: testSpaceFillingCurve.cpp
	$(CC) $(CFLAGS) -o testSpaceFillingCurve testSpaceFillingCurve.cpp

clean:
	rm -f testSpaceFillingCurve *.o
