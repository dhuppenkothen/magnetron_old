#include "Data.h"
#include <iostream>
#include <fstream>

using namespace std;

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

	double temp1; int temp2;
	while(fin>>temp1 && fin>>temp2)
	{
		t.push_back(temp1);
		y.push_back(temp2);
	}
	fin.close();

	if(t.size() == y.size())
		cout<<"# Found "<<t.size()<<" points in file "<<filename<<"."<<endl;
	else
		cerr<<"# There is a problem with the data in file "<<filename<<"."<<endl;
}

