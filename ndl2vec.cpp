#include <algorithm>
#include <atomic>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>


using CStr = char const *;

struct MemIO {
	size_t len;
	CStr   ptr;

	MemIO(std::string const & path) : len((size_t) -1), ptr(0) {
		int fd = ::open(path.c_str(), O_RDONLY);
		len = lseek(fd, 0, SEEK_END); lseek(fd, 0, SEEK_SET);
		ptr = (CStr) mmap(0, len, PROT_READ, MAP_PRIVATE, fd, 0);
		close(fd);
	}

	MemIO(MemIO && memio) : len(memio.len), ptr(memio.ptr) {
		memio.len = (size_t) -1;
		memio.ptr = 0;
	}

	~MemIO() {
		munmap((void *) ptr, len);
	}
};

using View = std::pair<CStr, CStr>;

std::string to_s(View view) {
	return std::string(view.first, view.second);
}

template <typename F>
void split(View view, char c, F f) {
	CStr p = view.first;
	CStr q = view.second;
	for (;;) {
		CStr r = std::find(p, q, c);
		if (r != p) {
			f(View(p, r));
		}
		if (r == q) {
			break;
		}
		p = r + 1;
	}
}


#define DIM 100

std::mt19937 mt(std::random_device{}());

std::uniform_real_distribution<double> unif(0, 1);

using ustring = std::basic_string<uint32_t>;

struct IDM {
	using S = std::string;

	std::vector<S>        texts;
	std::map<S, uint32_t> ids;

	IDM() : texts(), ids() {
		(*this)[S()];
	}

	uint32_t operator[](S const & text) {
		auto it = ids.find(text);
		if (it != ids.end()) {
			return it->second;
		}
		uint32_t id = (uint32_t) texts.size();
		texts.push_back(text);
		return ids[text] = id;
	}

	S operator[](ustring const & u) const {
		if (u.empty()) {
			return S();
		}
		S s(texts[u[0]]);
		for (size_t i = 1; i < u.length(); i++) {
			s += '\t';
			s += texts[u[i]];
		}
		return s;
	}
} idm;

struct Sigmoid {
	enum {
		SIZE   = 1024,
		WINDOW = 6,
	};

	double vs[SIZE];

	Sigmoid() : vs() {
		for (int i = 0; i < SIZE; i++) {
			// [0, SIZE) -> [-WINDOW, +WINDOW)
			double x = (2.0 * i / SIZE - 1) * WINDOW;
			vs[i] = std::exp(x) / (1 + std::exp(x));
		}
	}

	double operator()(double x) const {
		// [-WINDOW, +WINDOW) -> [0, SIZE)
		int i = (x / WINDOW + 1) / 2.0 * SIZE;
		return (i < 0 ? 0 : i < SIZE ? vs[i] : 1);
	}
} sigmoid;

struct Stat {
	uint32_t index;
	uint32_t freq;

	Stat() : index((uint32_t) -1), freq(0) {
	}
};

struct Layer {
	static constexpr size_t SIZE = DIM;

	float * ws; // do NOT free

	Layer(float * ws_) : ws(ws_) {
	}

	double operator*(Layer const & l) const {
		double sum = 0;
		for (size_t i = 0; i < SIZE; i++) {
			sum += ws[i] * l.ws[i];
		}
		return sum;
	}

	Layer & operator*=(double g) {
		for (size_t i = 0; i < SIZE; i++) {
			ws[i] *= g;
		}
		return *this;
	}

	void clear() {
		memset(ws, 0, sizeof(float) * SIZE);
	}

	void update(Layer const & l, double g) {
		for (size_t i = 0; i < SIZE; i++) {
			ws[i] += l.ws[i] * g;
		}
	}

	void write(FILE * fp) const {
		fwrite(ws, sizeof(float), SIZE, fp);
	}

	Layer(Layer const &) = delete;
	Layer & operator=(Layer const &) = delete;
};

struct Syn {
	static constexpr size_t SIZE = (Layer::SIZE + 7) & ~7; // round up to a multiple of 8

	size_t   len;
	float  * ws;

	Syn() : len(0), ws(0) {
	}

	~Syn() {
		free(ws);
	}

	void resize(size_t len_, bool random = 0) {
		free(ws);

		len = len_;
		posix_memalign((void **) &ws, 32, sizeof(float) * SIZE * len);
		if (random) {
			for (size_t i = 0; i < len; i++) {
				size_t j = 0;
				for (; j < Layer::SIZE; j++) {
					ws[SIZE * i + j] = (unif(mt) * 2 - 1) * 0.005;
				}
				for (; j <        SIZE; j++) { // for fraction
					ws[SIZE * i + j] = 0;
				}
			}
		} else {
			memset(ws, 0, sizeof(float) * SIZE * len);
		}
	}

	/**
	 * returns 32byte aligned memory
	 */
	float * operator[](size_t i) {
		return &ws[SIZE * i];
	}
};

using Disc = std::discrete_distribution<uint32_t>;

struct NN {
	static constexpr uint32_t MIN_FREQ = 3;
	static constexpr double   SAMPLE   = 1e-3;

	std::map<ustring, Stat> stats;
	size_t                  freq;

	Disc disc;
	Syn  syn0;
	Syn  syn1;

	Stat const * pstat(ustring const & u, int i, int n) const {
		auto it = stats.find(u.substr(i, n));
		return (it != stats.end() ? &it->second : 0);
	}

	uint32_t sample(Stat const & stat) const {
		double r = unif(mt);
		return (freq * SAMPLE < r * r * stat.freq ? (uint32_t) -1 : stat.index);
	}

	NN() : stats(), freq(0), disc(), syn0(), syn1() {
	}

	size_t setup() {
		freq = 0;

		std::vector<double> weights;
		for (auto it = stats.begin(); it != stats.end();) {
			Stat & stat = it->second;
			if (stat.freq < MIN_FREQ) {
				stats.erase(it++);
			} else {
				freq += stat.freq;
				stat.index = (uint32_t) weights.size();
				weights.push_back(std::pow(stat.freq, 0.75));
				++it;
			}
		}
		disc.param(Disc::param_type(weights.begin(), weights.end()));

		syn0.resize(stats.size());
		syn1.resize(stats.size(), 1);

		return freq;
	}

	void write(FILE * fp) {
		for (auto const & pair : stats) {
			ustring const & u    = pair.first;
			Stat    const & stat = pair.second;
			fprintf(fp, "%s ", idm[u].c_str());
			Layer(syn0[stat.index]).write(fp);
			fprintf(fp, "\n");
		}
	}
};

struct Model : private MemIO {
	static constexpr double ALPHA = 0.025;

	enum {
		N_GRAMS   = 5,
		N_EPOCHS  = 5,
		N_THREADS = 12,

		WINDOW    = 5,
		NEGATIVE  = 5,
	};

	std::vector<View>  sents;
	Syn                syn; // for working memory
	NN                 nns[N_GRAMS];

	double             alpha;
	std::atomic_size_t count;
	size_t             max_count;

	NN & get_nn(int n) {
		return nns[n - 1];
	}

	Model(std::string const & path) : MemIO(path),
		sents(), syn(), nns(),
		alpha(ALPHA), count(0), max_count(0) {
		split(View(ptr, ptr + len), '\n', [this] (View sent) {
			sents.push_back(sent);
		});
		size_t progress          = 0;
		size_t progress_per_kilo = 0;
		for (View sent : sents) {
			if (progress_per_kilo < progress / 1000) {
				progress_per_kilo = progress / 1000;
				printf("%ldK\r", progress_per_kilo); fflush(stdout);
			}
			ustring u;
			split(sent, ' ', [&u] (View view) {
				u += idm[to_s(view)];
			});
			int len = (int) u.length();
			for (int i = 0; i < len; i++) {
				for (int n = 1; n <= N_GRAMS; n++) {
					if (i + n > len) {
						break;
					}
					get_nn(n).stats[u.substr(i, n)].freq++;
					progress++;
				}
			}
		}
		for (NN & nn : nns) {
			max_count += nn.setup();
		}
		max_count *= N_EPOCHS;
	}

	void optimize(int t, uint32_t index0, uint32_t index1, Syn & syn0, Syn & syn1, Disc & disc, bool opposite = 0) {
		Layer l0(syn0[index1]);
		Layer neu1e(syn[t]);
		for (int d = 0; d < 1 + NEGATIVE; d++) {
			uint32_t index2;
			if (d == 0) {
				index2 = index0;
			} else {
				index2 = disc(mt);
				if (index2 == index0) {
					continue;
				}
			}
			Layer  l1(syn1[index2]);
			double x = l0 * l1;
			double g = ((opposite ? d != 0 : d == 0) - sigmoid(x)) * alpha;
			neu1e.update(l1, g);
			l1   .update(l0, g);
		}
		l0.update(neu1e, 1);
		neu1e.clear();
	}

	bool run(int t, ustring const & u, int i, int n0) {
		NN         & nn0 = get_nn(n0);
		Stat const * p0  = nn0.pstat(u, i, n0);
		if (!p0) {
			return 0;
		}
		uint32_t index0 = nn0.sample(*p0);
		if (index0 != (uint32_t) -1) {
			int len = (int) u.length();
			for (int n1 = 1; n1 <= N_GRAMS; n1++) {
				if (i      - n1 < 0) {
					break;
				}
				NN         & nn1 = get_nn(n1);
				Stat const * p1  = nn1.pstat(u, i - n1, n1);
				if (!p1) {
					continue;
				}
				uint32_t index1 = nn1.sample(*p1);
				if (index1 != (uint32_t) -1) {
					optimize(t, index0, index1, nn1.syn0, nn0.syn1, nn0.disc);
				}
			}
			for (int n1 = 1; n1 <= N_GRAMS; n1++) {
				if (i + n0 + n1 > len) {
					break;
				}
				NN         & nn1 = get_nn(n1);
				Stat const * p1  = nn1.pstat(u, i + n0, n1);
				if (!p1) {
					continue;
				}
				uint32_t index1 = nn1.sample(*p1);
				if (index1 != (uint32_t) -1) {
					optimize(t, index0, index1, nn1.syn0, nn0.syn1, nn0.disc);
				}
			}
		}
		return 1;
	}

	void start(int t) {
		size_t from = sents.size() * (t + 0) / N_THREADS;
		size_t to   = sents.size() * (t + 1) / N_THREADS;

		size_t c = 0;
		for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
			for (size_t index = from; index < to; index++) {
				if (c > 10000) {
					double r = (double) (count += c) / max_count; c = 0;
					alpha = ALPHA * std::max(1 - r, 0.0001);
					printf("\ralpha: %f  progress: %.2f%%", alpha, r * 100); fflush(stdout);
				}
				ustring u;
				split(sents[index], ' ', [&u] (View view) {
					u += idm[to_s(view)];
				});
				int len = (int) u.length();
				for (int i = 0; i < len; i++) {
					for (int n = 1; n <= N_GRAMS; n++) {
						if (i + n > len) {
							break;
						}
						c += run(t, u, i, n);
					}
				}
			}
		}
		double r = (double) (count += c) / max_count; c = 0;
		alpha = ALPHA * std::max(1 - r, 0.0001);
		printf("\ralpha: %f  progress: %.2f%%", alpha, r * 100); fflush(stdout);
	}

	void train() {
		syn.resize(N_THREADS);

		std::vector<std::thread> ths;
		for (int t = 0; t < N_THREADS; t++) {
			ths.push_back(std::thread(&Model::start, this, t));
		}
		for (auto & th : ths) {
			th.join();
		}
	}

	void save(char const * filename) {
		size_t voc = 0;
		for (NN & nn : nns) {
			voc += nn.stats.size();
		}

		FILE * fp = fopen(filename, "wb");
		fprintf(fp, "%ld %d\n", voc, DIM);
		for (NN & nn : nns) {
			nn.write(fp);
		}
		fclose(fp);
	}
};

int main(int argc, char ** argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: %s <INPUT> <MODEL>\n", argv[0]);
		fprintf(stderr, "argv[1]: input file name\n");
		fprintf(stderr, "argv[2]: model file name\n");
		exit(1);
	}

	Model model(argv[1]);
	model.train();
	model.save(argv[2]);

	return 0;
}
