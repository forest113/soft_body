#pragma once
#include <vector>
#include "GL/glut.h"
#include <chrono>
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
using namespace std;


#define PM_MASS 0.1      //mass of each vertex
#define SPRING_K  50 //spring constant
#define SPRING_DAMP 0.1 //damping force constant
#define TIME_STEP 0.0001
#define USE_R4 true

#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t code, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %d\n", cudaGetErrorString(code), line);
	}
}

/* A 3D vector class. */
class Vector3 {
public:
	double x;
	double y;
	double z;

	__device__ Vector3() {}

	__host__ __device__ Vector3(double x, double y, double z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}


	/* magnitude of the vector. */
	__device__ double mag() {
		return sqrt(pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2));
	}

	__device__ double dot(Vector3 v) {
		return (this->x * v.x + this->y * v.y + this->z * v.z);
	}

	__device__ void normalize() {
		double mag = this->mag();
		this->x /= mag;
		this->y /= mag;
		this->z /= mag;
	}

	__device__ void set_zero() {
		this->x = 0;
		this->y = 0;
		this->z = 0;
	}

	__device__ Vector3 operator * (double scalar) {
		return Vector3(this->x * scalar, this->y * scalar, this->z * scalar);
	}

	__device__ Vector3 operator / (double scalar) {
		return Vector3(this->x / scalar, this->y / scalar, this->z / scalar);
	}

	__device__ Vector3 operator - () {
		return Vector3(-this->x, -this->y, -this->z);
	}

	__device__ Vector3 operator + (Vector3 v) {
		return Vector3(this->x + v.x, this->y + v.y, this->z + v.z);
	}

	__device__ Vector3 operator - (Vector3 v) {
		return Vector3(this->x - v.x, this->y - v.y, this->z - v.z);
	}
};

/* A class to store vertices of the mesh. */
struct Point {
	Vector3 pos;
	bool is_fixed;
	Vector3 impulse;
	Vector3 vel;
	int num_springs;

	__host__ __device__ Point() {}

	__host__ __device__ Point(double px, double py, double pz) {
		this->pos = Vector3(px, py, pz);
		this->is_fixed = false;
		this->impulse = Vector3(0, 0, 0);
		this->vel = Vector3(0, 0, 0);
		this->num_springs = 0;
	}

};

/* function to get point - point distance. */
__host__ __device__ double distance(Point p1, Point p2) {
	double d = sqrt(pow(p2.pos.x - p1.pos.x, 2) +
		pow(p2.pos.y - p1.pos.y, 2) +
		pow(p2.pos.z - p1.pos.z, 2));
	return d;
}

/* Class to represent each of the springs between the vertices of the mesh. */
struct Spring {
	Point* p1;
	Point* p2;
	int ind1, ind2;
	double rest_len;
	double cur_len;

	Spring() {}

	Spring(Point* p1, Point* p2) {
		this->p1 = p1;
		this->p2 = p2;
		if (p1 && p2) {
			this->rest_len = distance(*p1, *p2);
			this->cur_len = this->rest_len;
		}
	}

	Spring(const Spring& s) {
		this->p1 = s.p1;
		this->p2 = s.p2;
		this->rest_len = s.rest_len;
		this->cur_len - s.cur_len;
	}
};
/*global variables to point to the simulation springs and vertices*/
 Point* G_points_list_d;
 Spring* G_springs_list_d;
 Point* G_points_list_h;
 Spring* G_springs_list_h;
 int G_num_points;
 int G_num_springs;

/**************************/
/* step funcctions*/
/**************************/






/**************************/
/* parallel step functions*/
/**************************/

/* reset impulse for each point to zero. */
__global__ void set_impulse_zero_kernel(Point* points, int num_points) {
	int i = threadIdx.x;
	if (i >= num_points) {
		return;
	}
	if (i < num_points) {
		points[i].impulse = Vector3(0, 0, 0);
		//printf("point pos: %f %f %f", points[i].pos.x, points[i].pos.y, points[i].pos.z);
	}
}

__global__ void fix_springs_pointers_kernel(Spring* springs_list, Point *points_list, int size) {
	int i = threadIdx.x;
	if (i >= size) {
		return;
	}
	springs_list[i].p1 = &(points_list[springs_list[i].ind1]);
	springs_list[i].p2 = &(points_list[springs_list[i].ind2]);
}

__global__ void process_springs_kernel(Spring* springs_list, Point*points_list, int size) {
	int i = threadIdx.x;
	if (i >= size) {
		return;
	}
	if (i < size) {
		
		Spring* spring = &(springs_list[i]);
		
		Point* p1 = &(points_list[spring->ind1]);
		Point* p2 = &(points_list[spring->ind2]);
		//printf("hi ");
		//printf("indices:%d %d", spring->ind1, spring->ind2);
		//printf("point pos: %d %d %d", p1->pos.x, p1->pos.y, p1->pos.z);
		spring->cur_len = distance(*p1, *p2);
		//printf("hi1 ");
		Vector3 spring_dir = Vector3(p1->pos.x - p2->pos.x, p1->pos.y - p2->pos.y, p1->pos.z - p2->pos.z);
		spring_dir.normalize();
		
		//cout << "p1 :" << p1->pos.x << " " << p1->pos.y << " " << p1->pos.z << endl;
		//cout << "p2 :" << p2->pos.x << " " << p2->pos.y << " " << p2->pos.z << endl;
		//cout << "spring dir:" << spring_dir.x << " " << spring_dir.y << " " << spring_dir.z<< endl;
		double x = spring->rest_len - spring->cur_len;
		double spring_impulse_mag = SPRING_K * x;
		float damping_mag = -SPRING_DAMP * (p1->vel.dot(spring_dir) - p2->vel.dot(spring_dir));
		spring_impulse_mag += damping_mag;
		//cout << "expanded" << endl;
		p1->impulse = p1->impulse + (spring_dir * spring_impulse_mag * TIME_STEP);
		p2->impulse = p2->impulse + (-spring_dir * spring_impulse_mag * TIME_STEP);

		//cout << "impulse:" << (-spring_dir * spring_impulse_mag).x << " " << (-spring_dir * spring_impulse_mag).y << " " << (-spring_dir * spring_impulse_mag).z << endl;
	}

}

__global__ void update_points_kernel(float DeltaTime, Point* points_list, int num_points) {
	int i = threadIdx.x;
	if (i >= num_points - 1) {
		return;
	}
	if (i < num_points) {
		Point* p = &(points_list[i]);
		if (p->is_fixed) {
			return;
		}
		/*update velocities*/
		if (p->impulse.mag() != 0.0) {
			p->vel = p->vel + (p->impulse / PM_MASS);
		}

		if (i == 5) {
			//cout << "impulse:" << p->impulse.x << " " << p->impulse.y << " " << p->impulse.z << endl;
			//cout << "vel" << p->vel.x << " " << p->vel.y << " " << p->vel.z << endl;
		}
		/* Euler integration for location*/
		//cout << "old:" << p->pos.x << " " << p->pos.y << " " << p->pos.z << endl;
		p->pos = p->pos + (p->vel * (DeltaTime));
		//cout << "new:" << p->pos.x << " " << p->pos.y << " " << p->pos.z << endl;

	}
}

void step(Point* points_list, int num_points, Spring* springs_list, int num_springs) {

	int block_size = 128;
	int n_points = num_points;
	int n_springs = num_springs;
	//cout << "num points:" << n_points << endl;
	set_impulse_zero_kernel <<< 1, 128 >>>(points_list, n_points);
	//cout << "fist kernel done" << endl;
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	//cout << "impulse:" << points_list[14].impulse.x << " " << points_list[14].impulse.y << " " << points_list[14].impulse.z << endl;
	//cout << "second start" << endl;
	process_springs_kernel <<<1, block_size>>> (springs_list, points_list, n_springs);
	
	//gpuErrchk(cudaPeekAtLastError());
	update_points_kernel <<<1, block_size>>> (TIME_STEP, points_list, n_points);
	//cout << "third end" << endl;
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaError_t err;
	err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

/**************************/
/* opengl render functions*/
/**************************/

void display()
{
	cudaMemcpy(G_points_list_h, G_points_list_d, G_num_points*sizeof(Point), cudaMemcpyDeviceToHost);
	cudaMemcpy(G_springs_list_h, G_springs_list_d, G_num_springs * sizeof(Spring), cudaMemcpyDeviceToHost);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	//cout << "numPOinnte 0: " << G_num_points << endl;
	Point* points_list = G_points_list_h;
	Spring* springs_list = G_springs_list_h;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, 1.0, 0.1, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(1.0, 0.0, 2.0,  // eye position
		0.0, 0.0, 0.0,  // look-at position
		0.0, 1.0, 0.0); // up direction
	glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_LINES);
	for (int i = 0; i < G_num_springs; i++)
	{
		glColor3f((float)((i + 1) / 10) * 3, 0.3, 0.3);
		Point* p1 = &(points_list[springs_list[i].ind1]);
		glVertex3f(p1->pos.x, p1->pos.y, p1->pos.z);
		Point* p2 = &(points_list[springs_list[i].ind2]);
		glVertex3f(p2->pos.x, p2->pos.y, p2->pos.z);
	}
	glEnd();
	glFlush();
}

void update(int value)
{
	auto start = std::chrono::high_resolution_clock::now();
	//cout << "num_points1: " << G_num_points << endl;
	for (int i = 0; i < (0.001 / TIME_STEP); i++) {
		step(G_points_list_d, G_num_points, G_springs_list_d, G_num_springs);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << value << "step duration:" << duration.count() / 1000 << std::endl;

	// Call the display function to redraw the updated mesh
	glutPostRedisplay();

	// Call the update function again after a certain delay
	glutTimerFunc(1, update, value + 1);
}

void test() {
	cout << "num_points3:" << G_num_points << endl;
}



