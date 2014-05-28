CFLAGS = -O2 -Wall -Wextra -ansi -pedantic
LIBS =  -lrjobject -ldnest3 -lgsl -lgslcblas -lboost_system -lboost_thread


default:
	g++ -O2 -Wall -Wextra -ansi -pedantic -I/home/daniela/sw/include/dnest3 -I/scratch/daniela/repositories/foreign/RJObject -c *.cpp
	g++ -o main *.o  -lrjobject -ldnest3 -lgsl -lgslcblas -lboost_system -lboost_thread -L/scratch/daniela/repositories/foreign/RJObject -L/home/daniela/sw/lib
	rm -f *.o


