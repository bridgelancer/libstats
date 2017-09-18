CXX = g++ -std=c++11
INCLUDES = -I/OLS -I/ADF
LDFLAGS = -larmadillo -llapack -lblas
CXXFLAGS = -O2

OBJS =

main: $(OBJS) main.cpp ADF/adf.cpp OLS/ols.cpp
	$(CXX)  -IOLS   -IADF   $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

inverse: inverse_check.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

vec:	VEC.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS) -ggdb

clean:
	rm -f *.o






