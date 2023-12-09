//project by Malika Iyer and Jordan Photis
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL

//make an acceleration matrix which is NUMENTITIES squared in size;
	
//vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
//vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);

__global__
void computePairwiseAccels(vector3 *d_accels, vector3 *d_hPos, double *mass){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
        int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k;
	
	vector3 distance;
	double magnitude_sq, magnitude, accelmag;

	//printf("mass: %f", mass[0]);
	//first compute the pairwise accelerations.  Effect is on the first argument.
	if(i < NUMENTITIES && j < NUMENTITIES){
		//for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(d_accels[i*NUMENTITIES+j],0,0,0); //i*NUMENTITIES so values can 
			}
			else{
				for (k=0;k<3;k++) {
					distance[k]= d_hPos[i][k]- d_hPos[j][k];
				}
				magnitude_sq = distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				magnitude = sqrt(magnitude_sq);
				//printf("magnitude: %f", magnitude);
				accelmag = -1 * GRAV_CONSTANT * mass[j]/magnitude_sq;
				FILL_VECTOR(d_accels[i*NUMENTITIES +j],
						accelmag*distance[0]/magnitude,	
						accelmag*distance[1]/magnitude,
						accelmag*distance[2]/magnitude);
			}
		//}
	}
}

__global__
void sumRowsandUpdate(vector3* d_accels, vector3* d_hVel, vector3* d_hPos, double* mass){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
        int j,k;
	vector3 accel_sum = {0,0,0};
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	
	if (i<NUMENTITIES){
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++){
				accel_sum[k]+=d_accels[i*NUMENTITIES + j][k];
			}
		}
		//d_accel_sum[i] = accel_sum;
		//compute the new velocity based on the acceleration and time interval
          	//compute the new position based on the velocity and time interval
		 for (k=0;k<3;k++){
                  	d_hVel[i][k] += accel_sum[k]*INTERVAL;
                  	d_hPos[i][k] += d_hVel[i][k]*INTERVAL;
          	}
	}
}

void compute(){
	//int blockSize = 16;
	//int numBlocks = (NUMENTITIES*NUMENTITIES + blockSize - 1)/blockSize;
	dim3 szBlk(16,16);
	dim3 nBlk((NUMENTITIES*NUMENTITIES +szBlk.x-1)/szBlk.x, (NUMENTITIES*NUMENTITIES +szBlk.y-1)/szBlk.y);
	int blockSize2 = 256;	
	int numBlocks2 = (NUMENTITIES*NUMENTITIES + blockSize2 - 1)/blockSize2;

	computePairwiseAccels<<<nBlk, szBlk>>>(d_accels, d_hPos, d_mass);
	sumRowsandUpdate<<<numBlocks2, blockSize2>>>(d_accels, d_hVel, d_hPos, d_mass);	
	
}






