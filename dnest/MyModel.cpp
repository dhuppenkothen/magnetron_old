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

const Data& MyModel::data = Data::get_instance();

MyModel::MyModel()
{

}

void MyModel::birth()
{
	if(num >= 10)
		return;

	position.push_back(data.get_t_min() + data.get_t_range()*randomU());
	amplitude.push_back(exp(log(1E-3) + log(1E3)*randomU()));
	width.push_back(log(1E-3*data.get_t_range()) + log(1E3)*randomU());

	num++;
}

void MyModel::death()
{
	if(num <= 0)
		return;

	int i = randInt(num);
	position.erase(position.begin() + i);


	num++;
}


void MyModel::fromPrior()
{
	int n = randInt(10);
	for(int i=0; i<n; i++)
		birth();
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

