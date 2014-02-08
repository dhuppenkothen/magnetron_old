#ifndef _Word_
#define _Word_

class Word
{
	private:
		double position;
		double log_amplitude;
		double log_width;
		double log_skew;

	public:
		Word();
		void from_prior();

};

#endif

