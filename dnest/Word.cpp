#include "Word.h"
#include "RandomNumberGenerator.h"
#include "Data.h"

using namespace std;

Word::Word()
{

}

void Word::from_prior()
{
	position = Data::get_instance().get_t_min() +
			Data::get_instance().get_t_range()*randomU();

	log_amplitude = log(1E-3*Data::get_instance().get_y_mean()) +
			log(1E6)*randomU();

	log_width = log(1E-3*Data::get_instance().get_t_range()) +
			log(1E3)*randomU();

	log_skew = log(0.1) + log(100.)*randomU();
}

