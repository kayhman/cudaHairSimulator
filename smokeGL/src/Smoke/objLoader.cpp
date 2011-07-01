#include "objLoader.h"
#include <fstream>
#include <algorithm>
#include "math.h"

ObjLoader::ObjLoader(const std::string& name)
{
	std::ifstream in(name.c_str());

	while(in.good())
	{
		char c;
		 in >> c;

		 if(c == 'v')
		 {
			 float x, y, z;
			 in  >> x;
			 in  >> y;
			 in  >> z;

			 vertices.push_back(x);
			 vertices.push_back(y);
			 vertices.push_back(z);
		 }
		 if(c == 'n')
		 {
			 float x, y, z;
			 in  >> x;
			 in  >> y;
			 in  >> z;

			 verticesNormal.push_back(x);
			 verticesNormal.push_back(y);
			 verticesNormal.push_back(z);
		 }

		 if(c == 'f')
		 {
			 int a, b, c;
			 int na, nb, nc;
			 char slash;
			 in  >> a; in >> slash; in >> slash; in  >> na;
			 in  >> b; in >> slash; in >> slash; in  >> nb;
			 in  >> c; in >> slash; in >> slash; in  >> nc;

			 triangles.push_back(a-1);
			 triangles.push_back(b-1);
			 triangles.push_back(c-1);

			 //triangleNormal.push_back(verticesNormal[(na-1)*3+0]);
			 //triangleNormal.push_back(verticesNormal[(na-1)*3+1]);
			 //triangleNormal.push_back(verticesNormal[(na-1)*3+2]);
		 }

	}

	//verticesNormal.resize(vertices.size());
	//std::fill(verticesNormal.begin(), verticesNormal.end(), 0.);
	//for(int i = 0 ; i < triangles.size()/3 ; i++)
	//{
	//	float Ax = vertices[3*triangles[3*i+0] + 0];
	//	float Ay = vertices[3*triangles[3*i+0] + 1];
	//	float Az = vertices[3*triangles[3*i+0] + 2];

	//	float Bx = vertices[3*triangles[3*i+1] + 0];
	//	float By = vertices[3*triangles[3*i+1] + 1];
	//	float Bz = vertices[3*triangles[3*i+1] + 2];

	//	float Cx = vertices[3*triangles[3*i+2] + 0];
	//	float Cy = vertices[3*triangles[3*i+2] + 1];
	//	float Cz = vertices[3*triangles[3*i+2] + 2];
	//		
	//	float ABx = Bx - Ax;
	//	float ABy = By - Ay;
	//	float ABz = Bz - Az;

	//	float ACx = Cx - Ax;
	//	float ACy = Cy - Ay;
	//	float ACz = Cz - Az;

	//	float nx = ACy * ABz - ABy * ACz;
	//	float ny = ACz * ABx - ABz * ACx;
	//	float nz = ACx * ABy - ABx * ACy;

	//	


	//	float norm = sqrt(nx * nx + ny * ny + nz * nz);

	//	nx /= norm;
	//	ny /= norm;
	//	nz /= norm;

	//	float nx0 = nx;
	//	float ny0 = ny;
	//	float nz0 = nz;

	//	/////////////////////////
	//	// normal at vertex a  //
	//	/////////////////////////
	//	nx = nx0; ny = ny0 ; nz = nz0;
	//	nx += verticesNormal[3*triangles[3*i+0] + 0];
	//	ny += verticesNormal[3*triangles[3*i+0] + 1];
	//	nz += verticesNormal[3*triangles[3*i+0] + 2];
	//	norm = sqrt(nx * nx + ny * ny + nz * nz);
	//	nx /= norm;
	//	ny /= norm;
	//	nz /= norm;

	//	verticesNormal[3*triangles[3*i+0] + 0] = nx;
	//	verticesNormal[3*triangles[3*i+0] + 1] = ny;
	//	verticesNormal[3*triangles[3*i+0] + 2] = nz;


	//	/////////////////////////
	//	// normal at vertex b  //
	//	/////////////////////////
	//	nx = nx0; ny = ny0 ; nz = nz0;
	//	nx += verticesNormal[3*triangles[3*i+1] + 0];
	//	ny += verticesNormal[3*triangles[3*i+1] + 1];
	//	nz += verticesNormal[3*triangles[3*i+1] + 2];
	//	norm = sqrt(nx * nx + ny * ny + nz * nz);
	//	nx /= norm;
	//	ny /= norm;
	//	nz /= norm;

	//	verticesNormal[3*triangles[3*i+1] + 0] = nx;
	//	verticesNormal[3*triangles[3*i+1] + 1] = ny;
	//	verticesNormal[3*triangles[3*i+1] + 2] = nz;


	//	/////////////////////////
	//	// normal at vertex c  //
	//	/////////////////////////
	//	nx = nx0; ny = ny0 ; nz = nz0;
	//	nx += verticesNormal[3*triangles[3*i+2] + 0];
	//	ny += verticesNormal[3*triangles[3*i+2] + 1];
	//	nz += verticesNormal[3*triangles[3*i+2] + 2];
	//	norm = sqrt(nx * nx + ny * ny + nz * nz);
	//	nx /= norm;
	//	ny /= norm;
	//	nz /= norm;

	//	verticesNormal[3*triangles[3*i+2] + 0] = nx;
	//	verticesNormal[3*triangles[3*i+2] + 1] = ny;
	//	verticesNormal[3*triangles[3*i+2] + 2] = nz;
	//}

	//for(int i = 0 ; i < verticesNormal.size() / 3 ; i++)
	//{
	//	float nx = verticesNormal[3 * i + 0];
	//	float ny = verticesNormal[3 * i + 1];
	//	float nz = verticesNormal[3 * i + 2];

	//	float norm = sqrt(nx * nx + ny * ny + nz * nz);

	//	nx /= norm;
	//	ny /= norm;
	//	nz /= norm;

	//	verticesNormal[3 * i + 0] = nx;
	//	verticesNormal[3 * i + 1] = ny;
	//	verticesNormal[3 * i + 2] = nz;

	//}
}
std::vector<float>& ObjLoader::getVertices()
{
	return vertices;
}

std::vector<float>& ObjLoader::getVerticesNormal()
{
	return verticesNormal;
}

std::vector<int>& ObjLoader::getTriangles()
{
	return triangles;
}

std::vector<float>& ObjLoader::getTrianglesNormal()
{
	return triangleNormal;
}