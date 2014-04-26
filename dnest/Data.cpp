#include "Data.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;

Data Data::instance;

Data::Data()
{

}

void Data::load(const char* filename)
{
	fstream fin(filename, ios::in);
	if(!fin)
	{
		cerr<<"# Failed to open file "<<filename<<"."<<endl;
		return;
	}

	t.clear();
	y.clear();

	double temp1, temp2;
	while(fin>>temp1 && fin>>temp2)
	{
		t.push_back(temp1);
		y.push_back(static_cast<int>(temp2));
	}
	fin.close();

	if(t.size() == y.size())
		cout<<"# Found "<<t.size()<<" points in file "<<filename<<"."<<endl;
	else
		cerr<<"# There is a problem with the data in file "<<filename<<"."<<endl;

	compute_summaries();
}

void Data::compute_summaries()
{
	t_min = *min_element(t.begin(), t.end());
	t_max = *max_element(t.begin(), t.end());
	t_range = t_max - t_min;

	y_mean = 0;
	for(size_t i=0; i<y.size(); i++)
		y_mean += y[i];
	y_mean /= y.size();
}

