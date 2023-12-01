#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// represents the objects in the system.  Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass, *d_mass;

//other global vars
size_t size_accels; //gets size of matrix and other values
size_t size_accel_sum;
size_t size_vel;
size_t size_pos;
size_t size_mass;

vector3 *d_accels;

//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
}

//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
}

//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES); //initializes memory for planets/asteroids
	planetFill(); //fills in data for planets
	randomFill(NUMPLANETS + 1, NUMASTEROIDS); //fills in data for asteroids
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout);
	printf("\n\n\n");
	#endif


	//memory allocation for cuda
	size_accels = sizeof(vector3)*NUMENTITIES*NUMENTITIES;
	size_vel = sizeof(vector3)*NUMENTITIES;
	size_pos = sizeof(vector3)*NUMENTITIES;
	size_mass = sizeof(double)*NUMENTITIES;

	//allocating memory for matrix in GPU
	cudaMalloc(&d_accels, size_accels);
	cudaMalloc(&d_hVel, size_vel);
	cudaMalloc(&d_hPos, size_pos);
	cudaMalloc(&d_mass, size_mass);

	
	//only need to copy data from host to device
	cudaMemcpy(d_hVel, hVel, size_vel, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hPos, hPos, size_pos, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, size_mass, cudaMemcpyHostToDevice);	
	
	
	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
		compute();
	}
    
	//after computation, copy data back from device to host
	cudaMemcpy(hVel, d_hVel, size_vel, cudaMemcpyDeviceToHost);
        cudaMemcpy(hPos, d_hPos, size_pos, cudaMemcpyDeviceToHost);
        //cudaMemcpy(mass, d_mass, size_mass, cudaMemcpyDeviceToHost);	


	clock_t t1=clock()-t0;
	#ifdef DEBUG
	printSystem(stdout);
	#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	//free matrix in GPU
	cudaFree(d_accels);
	cudaFree(d_hVel);
	cudaFree(d_hPos);
	cudaFree(d_mass);

	freeHostMemory();
}
