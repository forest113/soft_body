#include <iostream>
#include <math.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "soft_body_parallel.cuh"

extern  Point* G_points_list_d;
extern  Spring* G_springs_list_d;
extern int G_num_points;
extern int G_num_springs;

/* for hash map of edges. */
struct hashFunction
{
	size_t operator()(const vector<int>
		& myVector) const
	{
		std::hash<int> hasher;
		size_t answer = 0;

		for (int i : myVector)
		{
			answer ^= hasher(i) + 0x9e3779b9 +
				(answer << 6) + (answer >> 2);
		}
		return answer;
	}
};

int main(int argc, char** argv) {
	int num_points = 0, num_hex = 0;

	string c;
	ifstream fin;
	/* read the mesh. */
	fin.open("test.hex", ios::in);
	if (!fin) {
		std::cerr << "There was a problem opening the input file!\n";
		exit(1);//exit or do additional error checking
	}
	fin >> num_points;
	cout << num_points << endl;
	/* List of all the vertices. */
	vector<Point> points_list(num_points, Point(0, 0, 0));
	/* List of all the springs. */
	vector<Spring> springs_list;
	/* hashset of edges. */
	unordered_set<vector<int>, hashFunction> edges;
	/* read all points. */
	for (int i = 0; i < num_points; i++) {
		Point p;
		double px, py, pz;
		fin >> px >> py >> pz;
		p = Point(px, py, pz);
		points_list[i] = (p);
	}
	cout << "read" << points_list.size() << "points," << num_points << endl;;

	fin >> num_hex;
	cout << num_hex << endl;
	for (int i = 0; i < num_hex; i++) {
		int p[8] = { 0,0,0,0,0,0,0,0 }, ignore;
		fin >> p[0] >> p[1] >> p[2] >> p[3] >> p[4] >> p[5] >> p[6] >> p[7] >> ignore;
		/* create hashset containing pairs of vertices that form edges, so that springs do not repeat over eachother,
		 * in case of overlapping hexaderals. */
		vector<int> edge(2, 0);
		for (int i = 0; i < 4; i++) {
			/* this step adds edges 0-1, 1-2, 2-3, 3-0. */
			edge[0] = p[i];
			edge[1] = p[(i + 1) % 4];
			sort(edge.begin(), edge.end());
			if (edges.find(edge) == edges.end()) {
				edges.insert(edge);
			}
			/* this step adds edges 4-5, 5-6, 6-7, 7-8. */
			edge[0] = p[i + 4];
			edge[1] = p[((i + 1) % 4) + 4];
			sort(edge.begin(), edge.end());
			if (edges.find(edge) == edges.end()) {
				edges.insert(edge);
			}
			/* this steps adds edges 0-4, 1-5, 2-6, 3-7. */
			edge[0] = p[i];
			edge[1] = p[i + 4];
			sort(edge.begin(), edge.end());
			if (edges.find(edge) == edges.end()) {
				edges.insert(edge);
			}
		}
		edge.clear();
	}
	fin.close();

	cout << "num springs:" << edges.size() << endl;
	springs_list.resize(edges.size(), Spring(NULL, NULL));
	/* create springs from set of edges. */
	unordered_set<vector<int>>::iterator itr;
	int i = 0;
	for (itr = edges.begin();
		itr != edges.end(); itr++)
	{
		Point* p1, * p2;

		vector<int> edge = *itr;
		p1 = &points_list[edge[0] - 1];
		p2 = &points_list[edge[1] - 1];
		
		p1->num_springs += 1;
		p2->num_springs += 1;
		Spring s = Spring(p1, p2);
		s.ind1 = edge[0] - 1;
		s.ind2 = edge[1] - 1;
		springs_list[i] = (s);
		
		i++;
	}

	/* read input. */
	ifstream fin1;
	fin1.open("input.txt", ios::in);
	if (!fin1) {
		std::cerr << "There was a problem opening the input file!\n";
		exit(1);//exit or do additional error checking
	}
	int num_fixed = 0;
	//cout << "Enter number of fixed vertices:" << endl;
	fin1 >> num_fixed;
	for (int i = 0; i < num_fixed; i++) {
		int ind;
		//cout << "Enter " << i << "th" << "fixed point index" << endl;
		fin1 >> ind;
		points_list[ind].is_fixed = true;
	}

	if (num_fixed == 0) {
		//cout << "numspinrgs" << endl;
		for (int i = 0; i < points_list.size(); i++) {
			if (points_list[i].num_springs < 6) {
				points_list[i].is_fixed = true;
			}
		}cout << endl;
	}
	cout << "read something";

	int num_D_n = 0;
	//cout << "Enter number of vertices in D_n" << endl;
	fin1 >> num_D_n;
	for (int i = 0; i < num_D_n; i++) {
		int ind;
		//cout << "enter " << i << "th point in D_n";
		fin1 >> ind;
		Point* p = &points_list[ind];
		p->pos.x = p->pos.x + 0.2;
		//p->is_fixed = false;
	}

	cout << "simulating timesteps" << endl;
	//G_points_list = &points_list;
	cudaError_t cudaStatus = cudaMalloc(&G_points_list_d, points_list.size() * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
		// handle error condition
	}
	cudaMalloc(&G_springs_list_d, springs_list.size() * sizeof(Spring));
	cudaMemcpy(G_points_list_d, points_list.data(), points_list.size() * sizeof(Point), cudaMemcpyHostToDevice);
	//G_springs_list = &springs_list;
	cudaMemcpy(G_springs_list_d, springs_list.data(), springs_list.size() * sizeof(Spring), cudaMemcpyHostToDevice);
	G_num_points = points_list.size();
	cout << "num points:" << num_points << endl;
	
	G_num_springs = springs_list.size();
	G_points_list_h = (Point*)malloc(G_num_points * sizeof(Point));
	G_springs_list_h = (Spring*)malloc(G_num_springs * sizeof(Spring));
	fix_springs_pointers_kernel <<<1, 128 >>> (G_springs_list_d, G_points_list_d, G_num_springs);
	test();
	cout<<"yis"<<endl;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(500, 500);
	glutCreateWindow("OpenGL Window");
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glutDisplayFunc(display);
	glutTimerFunc(0, update, 0);
	glutMainLoop();

	cudaFree(G_points_list_d);
	cudaFree(G_springs_list_d);

	/*ofstream fout("sphere.obj");
	//cout << "points:" << endl;
	for (int i = 0; i < points_list.size(); i++) {
		fout << "v " << points_list[i].pos.x << " " << points_list[i].pos.y << " " << points_list[i].pos.z << endl;

	}
	for (itr = edges.begin();
		itr != edges.end(); itr++)
	{

		vector<int> edge = *itr;
		int p1 = edge[0];
		int p2 = edge[1];
		fout << "l " << p1 << " " << p2 << endl;
	}*/





}



