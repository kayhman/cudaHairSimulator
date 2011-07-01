#pragma once
#include <vector>

class SmokeRenderer
{
private:
	float *d_density;
	float *d_image;
	float *d_radiance;
	int gridX, gridY, gridZ;
	

	std::vector<float> density;
	std::vector<float> radiance;
	std::vector<float> image;


public:
	SmokeRenderer();
	~SmokeRenderer();
	void initSmoke();
	void render();


	const std::vector<float>& getDensity();
	const std::vector<float>& getRadiance();
	const std::vector<float>& getImage();
};