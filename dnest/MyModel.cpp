/*
* Copyright (c) 2009, 2010, 2011, 2012 Brendon J. Brewer.
*
* This file is part of DNest3.
*
* DNest3 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DNest3 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DNest3. If not, see <http://www.gnu.org/licenses/>.
*/

#include "MyModel.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include "Data.h"
#include <cmath>
#include <gsl/gsl_sf_gamma.h>

using namespace std;
using namespace DNest3;

const Data& MyModel::data = Data::get_instance();

MyModel::MyModel()
:bursts(2, 10, false, ClassicMassInf1D(data.get_t_min(), data.get_t_max(),
				1E-3*data.get_y_mean(), 1E3*data.get_y_mean()))
,mu(data.get_t().size())
{

}

void MyModel::calculate_mu()
{
	const vector< vector<double> >& components = bursts.get_components();
	const vector<double>& t = data.get_t();

	for(size_t i=0; i<mu.size(); i++)
	{
		mu[i] = 0.;
		for(size_t j=0; j<components.size(); j++)
		{
			mu[i] += components[j][1]
					*exp(-abs(t[i] - components[j][0])/10.);
		}
	}
}

void MyModel::fromPrior()
{
	bursts.fromPrior();
	calculate_mu();
}

double MyModel::perturb()
{
	double logH = 0.;

	logH += bursts.perturb();
	calculate_mu();

	return logH;
}

double MyModel::logLikelihood() const
{
	const vector<int>& y = data.get_y();

	double logl = 0.;
	for(size_t i=0; i<y.size(); i++)
		logl += -mu[i] + y[i]*log(mu[i]) - gsl_sf_lngamma(y[i] + 1.);

	return logl;
}

void MyModel::print(std::ostream& out) const
{
	bursts.print(out);
	for(size_t i=0; i<mu.size(); i++)
		out<<mu[i]<<' ';
}

string MyModel::description() const
{
	return string("");
}

