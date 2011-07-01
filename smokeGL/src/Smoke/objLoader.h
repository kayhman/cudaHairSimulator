#pragma once
#include <string>
#include <vector>

class ObjLoader
{
private:
	std::vector<int> triangles;
	std::vector<float> vertices;
	std::vector<float> verticesNormal;
	std::vector<float> triangleNormal;

public:
	ObjLoader(const std::string& name);

	std::vector<float>& getVertices();
	std::vector<float>& getVerticesNormal();
	std::vector<float>& getTrianglesNormal();
	std::vector<int>& getTriangles();

};