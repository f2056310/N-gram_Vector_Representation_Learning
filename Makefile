CXX = g++ -W -Wall -O3

BIN = ndl2vec

all: $(BIN)
clean:
	rm -rf $(BIN)

$(BIN): $(BIN).cpp
	$(CXX) -o $@ $^ -pthread
	 