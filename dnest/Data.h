#ifndef _Data_
#define _Data_

#include <vector>

class Data
{
	private:
		std::vector<double> t;
		std::vector<int> y;

		// Some useful summaries
		double t_min, t_max, t_range;
		double y_mean;
		void compute_summaries();

	public:
		Data();
		void load(const char* filename);

	// Singleton
	private:
		static Data instance;
	public:
		static Data& get_instance() { return instance; }
};

#endif

