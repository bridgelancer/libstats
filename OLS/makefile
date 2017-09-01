CXX = g++
INCLUDES =
LDFLAGS = -larmadillo -llapack -lblas
CXXFLAGS = -O2

OBJS =

main: $(OBJS) main.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

test: $(OBJS) test.cpp ols.h ols.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)


clean:
	rm -f *.o






