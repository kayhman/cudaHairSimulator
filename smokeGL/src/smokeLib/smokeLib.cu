#include "smokeLib.h"
#include <fstream>
#include <iostream>

#define M_PI 3.141516


__global__ void initSmokeK(float* density)
{
	int slice = blockIdx.y;
	int column = blockIdx.x;
	int row = threadIdx.x;

	int idx = slice * gridDim.x * gridDim.y + row * blockDim.x + column;
	density[idx] = column * 1.0 / blockDim.x;
}

__global__ void propagateLight(float* density, float* radiance,  float* image)
{
	float albedo = 1.0;
	float light = 1.0;
	float h = 0.1;

	int slice = blockIdx.x;
	int col = threadIdx.x;


	float tRay = 1.;
	for(int row = 0 ; row < 256 ; ++row)
	{
		int gidx = slice * gridDim.x * gridDim.y + row * blockDim.x + col;

		float tVox = exp(-0.3 * h * density[gidx]);
		tRay *= tVox;

		int ridx = slice * gridDim.x * gridDim.y + row + col * blockDim.x;
		radiance[gidx] = albedo * light * tRay ;
	}


	for(int row = 0 ; rcol < 256 ; ++row)
	{
		int gidx = slice * gridDim.x * gridDim.y + row * blockDim.x + col;

		float tVox = exp(-0.3 * h * density[gidx]);
		tRay *= tVox;

		int ridx = slice * gridDim.x * gridDim.y + row + col * blockDim.x;
		radiance[gidx] = albedo * light * tRay ;
	}

}

__global__ void renderImage(float* density, float* radiance, float* image)
{
}

SmokeRenderer::SmokeRenderer() :
d_density(NULL),
d_image(NULL),
gridX(256),
gridY(256),
gridZ(256)
{
	
}

SmokeRenderer::~SmokeRenderer()
{
	if(d_density)
		cudaFree(d_density);
	if(d_image)
		cudaFree(d_image);
}

void SmokeRenderer::initSmoke()
{
	dim3 gridSize(gridY, gridX); 
	dim3 blockSize(gridZ);

	int size = gridX * gridY * gridZ;
	int imageSize = gridX * gridY;


	/////////////////////////////////////////
	//            Init Position            //
	/////////////////////////////////////////
	cudaMalloc(&d_density, size * sizeof(float));
	cudaMalloc(&d_radiance, size * sizeof(float));
	cudaMalloc(&d_image, imageSize * sizeof(float));

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	initSmokeK<<<gridSize, blockSize >>>(d_density);
	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	std::ofstream out("log.smoke");
	out << "initSmoke : " << time << std::endl;
	out.close();

	density.resize(size);
	radiance.resize(size);
	image.resize(imageSize);
	
	cudaMemcpy(&(density[0]), d_density, size * sizeof(float), cudaMemcpyDeviceToHost);
}


void SmokeRenderer::render()
{
	dim3 gridSize(gridX); 
	dim3 blockSize(gridY);
	int size = gridX * gridY * gridZ;

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	propagateLight<<<gridSize, blockSize >>>(d_density, d_radiance);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );


	std::ofstream out("log.smoke", std::ios::app);
	out << "render : " << time << std::endl;
	out.close();

	std::cout << cudaGetErrorString(cudaGetLastError()) << std::cout;


	cudaMemcpy(&(radiance[0]), d_radiance, size * sizeof(float), cudaMemcpyDeviceToHost);

}


const std::vector<float>& SmokeRenderer::getDensity()
{
	return density;
}


const std::vector<float>& SmokeRenderer::getImage()
{
	return image;
}


const std::vector<float>& SmokeRenderer::getRadiance()
{
	return radiance;
}