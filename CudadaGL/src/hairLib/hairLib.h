#pragma once
#include <vector>

class HairSimulation
{
private:
	float *d_x, *d_y, *d_z;
	float *d_vx, *d_vy, *d_vz;
	float *d_gap, *d_velProj;
	float *d_dirX, *d_dirY, *d_dirZ;

	float x, y, z;
	float hxy, hz;
	float radius;
	int hairLenght;
	int gridX, gridY, gridZ;

	std::vector<float> X;
	std::vector<float> Y;
	std::vector<float> Z;




public:
	HairSimulation(float x, float y, float z, float radius, int gridX, int gridY, int hairLenght, float hxy, float hz);
	~HairSimulation();
	void initHair();
	void integrate(float dt);


	const std::vector<float>& getMassPositionX();
	const std::vector<float>& getMassPositionY();
	const std::vector<float>& getMassPositionZ();
};