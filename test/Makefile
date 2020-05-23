CC :=g++
#CC :=clang++-5.0
CXX_FLAGS +=-Wextra -Wall -Wpedantic -std=c++17 -march=native -Ofast -Wno-unknown-pragmas -Wno-attributes

GTEST_LIB :=-lgtest

#$(INTEL_INC)
INCS +=-I../include 
LIBS +=-lgtest -lpthread -lm

CFILES2 = $(wildcard src/*.cpp)
OBJS5   = $(notdir $(CFILES2))
OBJS6   = $(patsubst %.cpp,%.o,$(OBJS5))
OBJSC   = $(addprefix build/,$(OBJS6))


TARGET := main
$(TARGET): $(OBJSC)
	$(CC) $(OBJSC) $(LIBS) -o $(TARGET)

build/%.o : src/%.cpp
	$(CC) $(CXX_FLAGS) -c $(INCS)  $< -o $@

.PHONY : clean

clean:
	rm -f $(TARGET) build/*.* *debug*.m

