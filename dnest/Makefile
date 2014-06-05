CFLAGS = -O2 -Wall -Wextra -ansi -pedantic
LIBS =  -lrjobject -ldnest3 -lgsl -lgslcblas -lboost_system -lboost_thread


default:
    g++ -O2 -Wall -Wextra -ansi -pedantic -I/opt/local/include -I/usr/local/include/dnest3 -I/Users/daniela/repositories/foreign/RJObject -c *.cpp
    g++ -o main *.o -lrjobject -ldnest3 -lgsl -lgslcblas -lboost_system-mt -lboost_thread-mt -L/Users/daniela/repositories/foreign/RJObject -L/usr/local/lib -L/opt/local/lib 
    rm -f *.o




