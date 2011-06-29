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
	Z[idx] = Zi * hz ;
}

__global__ void applyGravity(float* X, float* Y, float*Z,
							 float* vx, float* vy, float* vz,
							 float dt)
{
	const float gravity = 9.81;
	const float mass = 1e-2;
	const float massInv = 1e2;
	const float alpha = 1.5 * mass; // Rayleigh damping


	int line = blockIdx.x;
	int Zi = blockIdx.y;

	int lineOffset = line * blockDim.x * blockDim.y;
	int Zoffset = Zi * (gridDim.x) * blockDim.x * blockDim.y;
	
	int idx = Zoffset + lineOffset + threadIdx.x;


	float Imass = -gravity * dt / massInv;
	float Idampx = - alpha * vx[idx] * dt;
	float Idampy = - alpha * vy[idx] * dt;
	float Idampz = - alpha * vz[idx] * dt;
	
	vx[idx] += Idampx / mass;
	vy[idx] += (Imass + Idampy) / mass;
	vz[idx] += Idampz / mass;
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

__device__ void computeConstraintImpulse(const float relX, const float relY, const float relZ,
									const float relvX, const float relvY, const float relvZ,
									const float dt, const float mass, const float massInv,
									const float refDist,
									float& changeX, float& changeY, float& changeZ)

{

	float dist = sqrt( relX * relX + relY * relY + relZ * relZ);
	float dx = relX / dist;
	float dy = relY / dist;
	float dz = relZ / dist;

	float velProj = dx * relvX + dy * relvY + dz * relvZ;
	float gap = dist - refDist ;

	float constI = (gap / dt + velProj) / massInv ;

	changeX = constI * dx / mass * (dist != 0.); 
	changeY = constI * dy / mass * (dist != 0.);
	changeZ = constI * dz / mass * (dist != 0.);
}

__device__ void computeBendingConstraintImpulse(const float relX, const float relY, const float relZ,
									const float relvX, const float relvY, const float relvZ,
									const float dt, const float mass, const float massInv,
									const float refDist,
									float& changeX, float& changeY, float& changeZ)

{
	float dist = sqrt( relX * relX + relY * relY + relZ * relZ);
	float dx = relX / dist;
	float dy = relY / dist;
	float dz = relZ / dist;

	float velProj = dx * relvX + dy * relvY + dz * relvZ;
	float gap = dist - refDist ;

	float constI = (gap / dt + velProj) / massInv ;

	changeX = constI * dx / mass * (gap < 0.) *  (dist != 0.); 
	changeY = constI * dy / mass * (gap < 0.) *  (dist != 0.);
	changeZ = constI * dz / mass * (gap < 0.) *  (dist != 0.);
}




__global__ void applyBothConstraint(float* X, float* Y, float*Z,
							 float* vx, float* vy, float* vz,
							 int hairLenght,
							 float hxy,
							 float hz,
							 float dt)
{

	const float mass = 1e-2;
	const float massInv = 1e2;

	int line = blockIdx.x;
	int lineOffset = line * blockDim.x * blockDim.y;

	
	{
		float massPositionXc = 0.;
		float massPositionYc = 0.;
		float massPositionZc = 0.;

		float  massVelocityXc = 0.;
		float  massVelocityYc = 0.;
		float  massVelocityZc = 0.;

		float massPositionXn = 0.;
		float massPositionYn = 0.;
		float massPositionZn = 0.;

		float  massVelocityXn = 0.;
		float  massVelocityYn = 0.;
		float  massVelocityZn = 0.;


		float massPositionXn1 = 0.;
		float massPositionYn1 = 0.;
		float massPositionZn1 = 0.;

		float  massVelocityXn1 = 0.;
		float  massVelocityYn1 = 0.;
		float  massVelocityZn1 = 0.;

		{
			int idxC = 0 + lineOffset + threadIdx.x ;
			massPositionXc = X[idxC];
			massPositionYc = Y[idxC];
			massPositionZc = Z[idxC];

			massVelocityXc = vx[idxC];
			massVelocityYc = vy[idxC];
			massVelocityZc = vz[idxC];

			// handle hair root
			float relX = threadIdx.x * hxy - massPositionXc;
			float relY = line * hxy - massPositionYc;
			float relZ = - massPositionZc;

			float relvX = - massVelocityXc;
			float relvY = - massVelocityYc;
			float relvZ = - massVelocityZc;


			float changeX, changeY, changeZ;
			computeConstraintImpulse(relX, relY, relZ,
				relvX, relvY, relvZ,
				dt, mass, massInv,
				0.,
				changeX, changeY,changeZ);

			vx[idxC] += changeX; 
			vy[idxC] += changeY;
			vz[idxC] += changeZ;

			massVelocityXc += changeX;
			massVelocityYc += changeY;
			massVelocityZc += changeZ;

		}

		{
			int idxC = gridDim.x * blockDim.x * blockDim.y + lineOffset + threadIdx.x ;
			massPositionXc = X[idxC];
			massPositionYc = Y[idxC];
			massPositionZc = Z[idxC];

			massVelocityXc = vx[idxC];
			massVelocityYc = vy[idxC];
			massVelocityZc = vz[idxC];

			// handle hair root
			float relX = threadIdx.x * hxy - massPositionXc;
			float relY = line * hxy - massPositionYc;
			float relZ = hz - massPositionZc;

			float relvX = - massVelocityXc;
			float relvY = - massVelocityYc;
			float relvZ = - massVelocityZc;


			float changeX, changeY, changeZ;
			computeConstraintImpulse(relX, relY, relZ,
				relvX, relvY, relvZ,
				dt, mass, massInv,
				0.,
				changeX, changeY,changeZ);

			vx[idxC] += changeX; 
			vy[idxC] += changeY;
			vz[idxC] += changeZ;

			massVelocityXc += changeX;
			massVelocityYc += changeY;
			massVelocityZc += changeZ;

		}

		for(int z = 0 ; z < hairLenght-2 ; ++z)
		{
			int ZoffC = z * (gridDim.x) * blockDim.x * blockDim.y;
			int ZoffN = (z+1) * (gridDim.x) * blockDim.x * blockDim.y;
			int ZoffN1 = (z+2) * (gridDim.x) * blockDim.x * blockDim.y;

			int idxC = ZoffC + lineOffset + threadIdx.x ;
			int idxN = ZoffN + lineOffset + threadIdx.x ;
			int idxN1 = ZoffN1 + lineOffset + threadIdx.x ;

			//Current particule
			massPositionXc = X[idxC];
			massPositionYc = Y[idxC];
			massPositionZc = Z[idxC];

			massVelocityXc = vx[idxC];
			massVelocityYc = vy[idxC];
			massVelocityZc = vz[idxC];

			//Next particule
			massPositionXn = X[idxN];
			massPositionYn = Y[idxN];
			massPositionZn = Z[idxN];

			massVelocityXn = vx[idxN];
			massVelocityYn = vy[idxN];
			massVelocityZn = vz[idxN];

			//Following Next particule
			massPositionXn1 = X[idxN1];
			massPositionYn1 = Y[idxN1];
			massPositionZn1 = Z[idxN1];

			massVelocityXn1 = vx[idxN1];
			massVelocityYn1 = vy[idxN1];
			massVelocityZn1 = vz[idxN1];


			////////////////////////////////////////////
			//           Apply stretch constraint     //
			////////////////////////////////////////////
			float relX = massPositionXn - massPositionXc;
			float relY = massPositionYn - massPositionYc;
			float relZ = massPositionZn - massPositionZc;


			float relvX = massVelocityXn - massVelocityXc;
			float relvY = massVelocityYn - massVelocityYc;
			float relvZ = massVelocityZn - massVelocityZc;

			float changeX, changeY, changeZ;
			computeConstraintImpulse(relX, relY, relZ,
				relvX, relvY, relvZ,
				dt, mass, massInv + massInv * (z != 1),
				hz,
				changeX, changeY,changeZ);

			vx[idxC] += changeX * (z != 1); 
			vy[idxC] += changeY * (z != 1);
			vz[idxC] += changeZ * (z != 1);

			vx[idxN] += -changeX; 
			vy[idxN] += -changeY;
			vz[idxN] += -changeZ;

			////////////////////////////////////////////
			//           Apply bending constraint     //
			////////////////////////////////////////////
			relX = massPositionXn1 - massPositionXc;
			relY = massPositionYn1 - massPositionYc;
			relZ = massPositionZn1 - massPositionZc;


			relvX = massVelocityXn1 - massVelocityXc;
			relvY = massVelocityYn1 - massVelocityYc;
			relvZ = massVelocityZn1 - massVelocityZc;

			computeConstraintImpulse(relX, relY, relZ,
				relvX, relvY, relvZ,
				dt, mass, massInv + massInv * (z != 1),
				2*hz,
				changeX, changeY,changeZ);

			vx[idxC] += changeX * (z != 1); 
			vy[idxC] += changeY * (z != 1);
			vz[idxC] += changeZ * (z != 1);

			vx[idxN1] += -changeX; 
			vy[idxN1] += -changeY;
			vz[idxN1] += -changeZ;
		}
	}
}


HairSimulation::HairSimulation(float x, float y, float z, float radius, int gridX, int gridY, int hairLenght, float hxy, float hz) :
x(x),
y(y), 
z(z),
radius(radius),
hairLenght(hairLenght),
d_x(NULL),
d_y(NULL),
d_z(NULL),
hxy(hxy),
hz(hz),
gridX(gridX),
gridY(gridY)
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
	dim3 gridSize(gridY, hairLenght); 
	dim3 blockSize(gridX, 1);

	int size = gridSize.x * gridSize.y * blockSize.x * blockSize.y;


	/////////////////////////////////////////
	//            Init Position            //
	/////////////////////////////////////////
	cudaMalloc(&d_x, size * sizeof(float));
	cudaMalloc(&d_y, size * sizeof(float));
	cudaMalloc(&d_z, size * sizeof(float));

	cudaMalloc(&d_gap, size * sizeof(float));
	cudaMalloc(&d_velProj, size * sizeof(float));

	cudaMalloc(&d_dirX, size * sizeof(float));
	cudaMalloc(&d_dirY, size * sizeof(float));
	cudaMalloc(&d_dirZ, size * sizeof(float));


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
	dim3 gridSize(gridY, hairLenght);
	dim3 blockSize(gridX, 1);
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
	
	for(int nbIter = 0 ; nbIter < 5 ; ++ nbIter)
	{
		/////////////////////////////////////////
		//   Integrate Constrained Motion      //
		/////////////////////////////////////////
		dim3 gridSize2(gridX, 1);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );

		applyBothConstraint<<<gridSize2, blockSize >>>(d_x, d_y, d_z, 
			d_vx, d_vy, d_vz,
			hairLenght,
			hxy,
			hz,
			dt);

		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );
		out << cudaGetErrorString(cudaGetLastError()) << std::endl;
		out << "constraint : " << time << std::endl;

		/////////////////////////////////////////////////
		//   Integrate Bending Constrained Motion      //
		/////////////////////////////////////////////////
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );

		//applyBendingConstraint<<<gridSize2, blockSize >>>(d_x, d_y, d_z, 
		//	d_vx, d_vy, d_vz,
		//	hairLenght,
		//	hxy,
		//	hz,
		//	dt);

		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );
		out << cudaGetErrorString(cudaGetLastError()) << std::endl;
		out << "bending constraint : " << time << std::endl;
	}
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