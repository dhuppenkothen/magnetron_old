#ifndef _ClassicMassInf1D_
#define _ClassicMassInf1D_

#include "Distributions/Distribution.h"

class ClassicMassInf1D:public Distribution
{
	private:
		// Limits
		double x_min, x_max;
		double mu_min, mu_max;

		// Mean of amplitudes and widths
		double mu, mu_widths;

		// Uniform for skews
		double a, b, k; // Endpoints are (a, b), parameterise with (b, k=a/b)

		double perturb_parameters();

	public:
		ClassicMassInf1D(double x_min, double x_max,
					double mu_min, double mu_max);

		void fromPrior();

		double log_pdf(const std::vector<double>& vec) const;
		void from_uniform(std::vector<double>& vec) const;
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;
};

#endif

