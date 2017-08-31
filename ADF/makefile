CXX = g++
INCLUDES = 
LDFLAGS = -larmadillo -llapack -lblas
CXXFLAGS = -O2 -std=c++11

OBJS =

main: $(OBJS) main.cpp ols.cpp adf.cpp
	$(CXX) $^ $(INCLUDES) -o $@ $(CXXFLAGS) $(LDFLAGS)

test: $(OBJS) test.cpp ols.h ols.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)


clean:
	rm -f *.o






