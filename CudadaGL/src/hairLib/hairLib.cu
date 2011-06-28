#include "hairLib.h"
#include <fstream>

__global__ void initHairs(float* X, float* Y, float*Z, float hxy, float hz)
{
	int line = blockIdx.x;
	int Zi = blockIdx.y;

	int lineOffset = line * blockDim.x * blockDim.y;
	int Zoffset = Zi * (gridDim.x) * blockDim.x * blockDim.y;
	
	int idx = Zoffset + lineOffset + threadIdx.x ;
	X[idx] = threadIdx.x * hxy;
	Y[idx] = line * hxy;
	Z[idx] = (Zi-1) * hz * (Zi > 1); // Two first position are in Z = 0;

	//int slice = blockIdx.y;
	//int Zi = threadIdx.x;
	//int hairIdInLine = blockIdx.x * blockDim.y + threadIdx.y;

	//int idx = slice * blockDim.x * blockDim.y * gridDim.x + hairIdInLine * blockDim.x + threadIdx.x;

	//X[idx] = hairIdInLine * hxy;
	//Y[idx] = slice * hxy;
	//Z[idx] = Zi * hz;
}

__global__ void applyGravity(float* X, float* Y, float*Z,
							 float* vx, float* vy, float* vz,
							 float dt)
{
	const float gravity = 9.81;
	const float mass = 1e-5;
	const float massInv = 1e5;
	const float alpha = 0.5 * mass; // Rayleigh damping


	int line = blockIdx.x;
	int Zi = blockIdx.y;

	int lineOffset = line * blockDim.x * blockDim.y;
	int Zoffset = Zi * (gridDim.x) * blockDim.x * blockDim.y;
	
	int idx = Zoffset + lineOffset + threadIdx.x;


	float Imass = -gravity * dt / massInv;
	float Idamp = - alpha * mass * dt;
	
	vy[idx] += (Imass + Idamp) / mass * (Zi > 0);
}

__global__ void integrateK(float* X, float* Y, float*Z,
						  float* vx, float* vy, float* vz,
						  float dt)
{
	int line = blockIdx.x;
	int Zi = blockIdx.y;

	int lineOffset = line * blockDim.x * blockDim.y;
	int Zoffset = Zi * (gridDim.x) * blockDim.x * blockDim.y;
	
	int idx = Zoffset + lineOffset + threadIdx.x ;

	// Integrate velocity
	X[idx] += vx[idx] * dt;
	Y[idx] += vy[idx] * dt;
	Z[idx] += vz[idx] * dt;
}

__global__ void applyConstraint(float* X, float* Y, float*Z,
							 float* vx, float* vy, float* vz,
							 float hz,
							 float dt)
{
	const float mass = 1e-5;
	const float massInv = 1e5;

	__shared__ float  massPositionX[256*2];
	__shared__ float  massPositionY[256*2];
	__shared__ float  massPositionZ[256*2];

	__shared__ float  massVelocityX[256*2];
	__shared__ float  massVelocityY[256*2];
	__shared__ float  massVelocityZ[256*2];


	int line = blockIdx.x;
	int lineOffset = line * blockDim.x * blockDim.y;
	
	int depth = 30;
	for(int z = 0 ; z < depth-1 ; ++z)
	{
		int ZoffC = z * (gridDim.x) * blockDim.x * blockDim.y;
		int ZoffN = (z+1) * (gridDim.x) * blockDim.x * blockDim.y;


		int idxC = ZoffC + lineOffset + threadIdx.x ;
		int idxN = ZoffN + lineOffset + threadIdx.x ;


		massPositionX[threadIdx.x] = X[idxC];
		massPositionY[threadIdx.x] = Y[idxC];
		massPositionZ[threadIdx.x] = Z[idxC];

		massVelocityX[threadIdx.x] = vx[idxC];
		massVelocityY[threadIdx.x] = vy[idxC];
		massVelocityZ[threadIdx.x] = vz[idxC];

		__syncthreads();
		
		massPositionX[256 + threadIdx.x] = X[idxN];
		massPositionY[256 + threadIdx.x] = Y[idxN];
		massPositionZ[256 + threadIdx.x] = Z[idxN];

		massVelocityX[256 + threadIdx.x] = vx[idxN];
		massVelocityY[256 + threadIdx.x] = vy[idxN];
		massVelocityZ[256 + threadIdx.x] = vz[idxN];

		__syncthreads();

		const float relX = massPositionX[256 + threadIdx.x] - massPositionX[threadIdx.x];
		const float relY = massPositionY[256 + threadIdx.x] - massPositionY[threadIdx.x];
		const float relZ = massPositionZ[256 + threadIdx.x] - massPositionZ[threadIdx.x];


		const float relvX = massVelocityX[256 + threadIdx.x] - massVelocityX[threadIdx.x];
		const float relvY = massVelocityY[256 + threadIdx.x] - massVelocityY[threadIdx.x];
		const float relvZ = massVelocityZ[256 + threadIdx.x] - massVelocityZ[threadIdx.x];


		float dist = sqrt( relX * relX + relY * relY + relZ * relZ);
		if(dist != 0.)
		{
			const float dx = relX / dist;
			const float dy = relY / dist;
			const float dz = relZ / dist;

			const float velProj = dx * relvX + dy * relvY + dz * relvZ;
			const float gap = dist - hz * (z > 0);

			const float constI = (gap / dt + velProj) / (massInv + massInv * (z > 0));

			vx[idxC] += constI * dx / mass*(z > 0); 
			vy[idxC] += constI * dy / mass*(z > 0);
			vz[idxC] += constI * dz / mass*(z > 0);

			vx[idxN] += -constI * dx / mass; 
			vy[idxN] += -constI * dy / mass;
			vz[idxN] += -constI * dz / mass;
		}
	}
	//dist = self.mass2.pos - self.mass1.pos;
	//if linalg.norm(dist) != 0.:
	//distUnit = dist / linalg.norm(dist);

	//relVel = dot((self.mass2.vel - self.mass1.vel), distUnit);
	//relDist = linalg.norm(dist) - self.refDist;

	//I = (relDist / dt + relVel) / (self.mass1.iMass + self.mass2.iMass) * distUnit;
	//self.mass1.applyImpulse(I);
	//self.mass2.applyImpulse(-I);
}

HairSimulation::HairSimulation(float x, float y, float z, float radius, int nbPartByHair, float hxy, float hz) :
x(x),
y(y), 
z(z),
radius(radius),
nbHairs(nbHairs),
blockX(256),
blockY(1),
d_x(NULL),
d_y(NULL),
d_z(NULL),
hxy(hxy),
hz(hz)
{
	
}

HairSimulation::~HairSimulation()
{
	if(d_x)
		cudaFree(d_x);
	if(d_y)
		cudaFree(d_y);
	if(d_z)
		cudaFree(d_z);
}

void HairSimulation::initHair()
{
	dim3 gridSize(256, 30); 
	dim3 blockSize(blockX, blockY);

	int size = gridSize.x * gridSize.y * blockSize.x * blockSize.y;


	/////////////////////////////////////////
	//            Init Position            //
	/////////////////////////////////////////
	cudaMalloc(&d_x, size * sizeof(float));
	cudaMalloc(&d_y, size * sizeof(float));
	cudaMalloc(&d_z, size * sizeof(float));


	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	initHairs<<<gridSize, blockSize >>>(d_x, d_y, d_z, hxy, hz);
	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	std::ofstream out("log.cudada");
	out << "initHairs : " << time << std::endl;
	out.close();

	X.resize(size);
	Y.resize(size);
	Z.resize(size);

	cudaMemcpy(&(X[0]), d_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&(Y[0]), d_y, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&(Z[0]), d_z, size * sizeof(float), cudaMemcpyDeviceToHost);
	/////////////////////////////////////////
	//            Init Velocity            //
	/////////////////////////////////////////
	cudaMalloc(&d_vx, size * sizeof(float));
	cudaMalloc(&d_vy, size * sizeof(float));
	cudaMalloc(&d_vz, size * sizeof(float));

	cudaMemset(d_vx, 0, sizeof(float));
	cudaMemset(d_vy, 0, sizeof(float));
	cudaMemset(d_vz, 0, sizeof(float));


}

void HairSimulation::integrate(float dt)
{
	dim3 gridSize(256, 30);
	dim3 blockSize(blockX, blockY);
	int size = gridSize.x * gridSize.y * blockSize.x * blockSize.y;

	/////////////////////////////////////////
	//        Integrate Free Motion        //
	/////////////////////////////////////////
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	applyGravity<<<gridSize, blockSize >>>(d_x, d_y, d_z, 
		d_vx, d_vy, d_vz,
		dt);
	
	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	std::ofstream out("log.cudada", std::ios::app);
	out << "gravity : " << time << std::endl;
	

	/////////////////////////////////////////
	//   Integrate Constrained Motion      //
	/////////////////////////////////////////
	dim3 gridSize2(256, 1);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	applyConstraint<<<gridSize2, blockSize >>>(d_x, d_y, d_z, 
		d_vx, d_vy, d_vz,
		hz,
		dt);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	out << cudaGetErrorString(cudaGetLastError()) << std::endl;
	out << "constraint : " << time << std::endl;

	/////////////////////////////////////////
	//            Time integration         //
	/////////////////////////////////////////
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
	integrateK<<<gridSize, blockSize >>>(d_x, d_y, d_z, 
		d_vx, d_vy, d_vz,
		dt);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	out << "integrate : " << time << std::endl;
	out.close();
	/////////////////////////////////////////
	//             Transfter data          //
	/////////////////////////////////////////
	cudaMemcpy(&(X[0]), d_x, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&(Y[0]), d_y, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&(Z[0]), d_z, size * sizeof(float), cudaMemcpyDeviceToHost);
}


const std::vector<float>& HairSimulation::getMassPositionX()
{
	return X;
}


const std::vector<float>& HairSimulation::getMassPositionY()
{
	return Y;
}


const std::vector<float>& HairSimulation::getMassPositionZ()
{
	return Z;
}