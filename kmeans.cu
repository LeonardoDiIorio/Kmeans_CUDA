
#include<cuda.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>


using namespace std;

static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


static void CheckCudaErrorAux(const char* file, unsigned line,
    const char* statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
        << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}


__global__ void kmeansIterationKernel(float* points, float* centroids, float* newCentroids, int* membersCounter, int dim, int numPoints, int numClusters)
{
    
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    if (t < numPoints)
    {
        int nearestCluster = 0;
        int minDistance = 100000000;

        //determina il centroide
        for (int c = 0; c < numClusters; c++)
        {
            float euclideanDistance = 0;
            for (int d = 0; d < dim; d++)
            {
                euclideanDistance += pow(*(points + t * dim + d) - *(centroids + c * dim + d), 2);
            }
            euclideanDistance = sqrt(euclideanDistance);
            if (euclideanDistance < minDistance)
            {
                minDistance = euclideanDistance;
                nearestCluster = c;
            }
        }

        for (int j = 0; j < dim; j++)
        {
            *(newCentroids + nearestCluster * dim + j) += *(points + t * dim + j);
        }
        //mantiene il numero di punti in ciascun cluster
        //aggiungere atomic
        atomicAdd(&membersCounter[nearestCluster], 1);
      

        __threadfence();

        if (t < numClusters)
        {
            for (int i = 0; i < numClusters; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    *(newCentroids + i * dim + j) = (*(newCentroids + i * dim + j)) / membersCounter[i];
                }
            }

            __threadfence();

            for (int i = 0; i < numClusters; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    *(centroids + i * dim + j) = *(newCentroids + i * dim + j);
                }
            }
        }
    }
}


//gestisce le iterazioni
void kmeans(float* devicePoints, float* deviceCentroids, int numPoints, int numClusters, int dim, int MAX_ITER, int blDim)
{
    float* newDeviceCentroids;
    int* deviceMemberCounters;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&newDeviceCentroids, numClusters * dim * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceMemberCounters, numClusters * sizeof(int)));
    for (int i = 0; i < MAX_ITER; i++)
    {
        float* newCentroids = new float[numClusters * dim] {0};
        int* membersCounter = new int[numClusters] {0};
        CUDA_CHECK_RETURN(cudaMemcpy((void*)deviceMemberCounters, (void*)membersCounter, numClusters * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy((void*)newDeviceCentroids, (void*)newCentroids, numClusters * dim * sizeof(float), cudaMemcpyHostToDevice));     
        kmeansIterationKernel << <ceil(numPoints / (float)blDim), blDim >> > (devicePoints, deviceCentroids, newDeviceCentroids, deviceMemberCounters, dim, numPoints, numClusters);
    

    }
    CUDA_CHECK_RETURN(cudaFree(newDeviceCentroids));
    CUDA_CHECK_RETURN(cudaFree(deviceMemberCounters));
}


int main(int argc, char** argv)
{
    int MAX_ITER = 40;
    int blDim = 1024;
    int dim = 3;
    int numPoints = 100000;
    int numClusters = 316;
    float* points = new float[numPoints * dim]; //matrice dei punti
    float* centroids = new float[numClusters * dim]; //matrice dei centroidi
    float* devicePoints;
    float* deviceCentroids;

    ofstream myfile;
    myfile.open("kmeans.csv");

    for (int k = 0; k < 10; k++)
    {
        uniform_real_distribution<float> unif(0, 1000);
        default_random_engine re;

        


        //inizializzo la matrice dei punti
        for (int i = 0; i < numPoints; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                *(points + i * dim + j) = unif(re);
            }
        }

        //inizializzo la matrice dei centroidi
        for (int h = 0; h < numClusters; h++)
        {
            int randomPoint = rand() % numClusters;
            for (int d = 0; d < dim; d++)
            {
                *(centroids + h * dim + d) = *(points + randomPoint * dim + d);
            }
        }

        //alloco spazio in memoria sul device
        CUDA_CHECK_RETURN(cudaMalloc((void**)&devicePoints, numPoints * dim * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceCentroids, numClusters * dim * sizeof(float)));
        // CUDA_CHECK_RETURN(cudaMalloc((void **) &deviceMemberCounters, numClusters * sizeof(int)));

         //copio dati sul device
        CUDA_CHECK_RETURN(cudaMemcpy((void*)devicePoints, (void*)points, numPoints * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy((void*)deviceCentroids, (void*)centroids, numClusters * dim * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK_RETURN(cudaMemcpy((void *) deviceMemberCounters, (void *) membersCounter, numClusters * sizeof(int), cudaMemcpyHostToDevice));

        int time = 0;
        auto start = std::chrono::system_clock::now();
        kmeans(devicePoints, deviceCentroids, numPoints, numClusters, dim, MAX_ITER, blDim);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        time += elapsed.count();
        std::cout << "\t\tTime: " << time << std::endl;
        myfile << time << "\n";
        CUDA_CHECK_RETURN(cudaFree(devicePoints));
        CUDA_CHECK_RETURN(cudaFree(deviceCentroids));
    }
    
    myfile.close();
}
