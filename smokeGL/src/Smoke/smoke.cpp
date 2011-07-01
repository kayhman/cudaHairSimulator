#include "smokelib.h"
#include "Cimg.h"

using namespace cimg_library;


int main()
{
	SmokeRenderer smoke;

	smoke.initSmoke();
	smoke.render();

	std::vector<float> density = smoke.getDensity();
	std::vector<float> radiance = smoke.getRadiance();

	CImg<unsigned char> visu(256,256,1, 1);
	CImgDisplay draw_disp(visu,"smoke");

	

	int t = 0;
	while (!draw_disp.is_closed()) 
	{
		int s = t % 256;
		smoke.render();
		for(int i = 0 ; i < 256 ; ++i)
		{
			for(int j = 0 ; j < 256 ; ++j)
			{
				unsigned char tmp = density[i + j* 256* 256 + s * 256] * 255.0;
				visu(j, i, 0) = tmp;
			}
		}
		//visu(128, 128, 0) = 1.0;
		draw_disp.display(visu);
		t++;
	}

	return 0;
}