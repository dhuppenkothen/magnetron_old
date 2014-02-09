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

using namespace std;
using namespace DNest3;

MyModel::MyModel()
:data(Data::get_instance())
{

}

void MyModel::birth()
{

}

void MyModel::fromPrior()
{
	num = randInt(10);
	position.resize(num);
	amplitude.resize(num);
	width.resize(num);
	for(int i=0; i<num; i++)
	{
		position[i] = data.get_t_min() + data.get_t_range()*randomU();
		amplitude[i] = exp(log(1E-3) + log(1E3)*randomU());
		width[i] = log(1E-3*data.t_range()) + log(1E3)*randomU();
	}
}

double MyModel::perturb()
{
	return 0.;
}

double MyModel::logLikelihood() const
{
	return 0.;
}

void MyModel::print(std::ostream& out) const
{

}

string MyModel::description() const
{
	return string("");
}

