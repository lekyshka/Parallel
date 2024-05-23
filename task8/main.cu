#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
namespace po = boost::program_options;

#include <cuda_runtime.h>
#include <cub/cub.cuh>

// cuda unique_ptr
template<typename T>
using cuda_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;

// new
template<typename T>
T* cuda_new(size_t size)
{
    T *d_ptr;
    cudaError_t cudaErr = cudaMalloc((void **)&d_ptr, sizeof(T) * size);
	if (cudaErr != cudaSuccess ){
		std::cerr << "Memory transfering error" << std::endl;
		exit(3);
	}
    return d_ptr;
}

// delete
template<typename T>
void cuda_delete(T *dev_ptr)
{
    cudaFree(dev_ptr);
}


cudaStream_t* cuda_new_stream()
{
    cudaStream_t* stream = new cudaStream_t;
    cudaStreamCreate(stream);
    return stream;
}

void cuda_delete_stream(cudaStream_t* stream)
{
    cudaStreamDestroy(*stream);
    delete stream;
}

cudaGraph_t* cuda_new_graph()
{
    cudaGraph_t* graph = new cudaGraph_t;
    return graph;
}

void cuda_delete_graph(cudaGraph_t* graph)
{
    cudaGraphDestroy(*graph);
    delete graph;
}

cudaGraphExec_t* cuda_new_graph_save()
{
    cudaGraphExec_t* graphExec = new cudaGraphExec_t;
    return graphExec;
}

void cuda_delete_graph_save(cudaGraphExec_t* graphExec)
{
    cudaGraphExecDestroy(*graphExec);
    delete graphExec;
}



void initialize(std::unique_ptr<double[]> &A, std::unique_ptr<double[]> &Anew, int n)
{
    memset(A.get(), 0, n * n * sizeof(double));

    double corners[4] = {10, 20, 30, 20};
    A[0] = corners[0];
    A[n - 1] = corners[1];
    A[n * n - 1] = corners[2];
    A[n * (n - 1)] = corners[3];
    double step = (corners[1] - corners[0]) / (n - 1);


    for (int i = 1; i < n - 1; i ++) {
        A[i] = corners[0] + i * step;
        A[n * i] = corners[0] + i * step;
        A[(n-1) + n * i] = corners[1] + i * step;
        A[n * (n-1) + i] = corners[3] + i * step;
    }
    std::memcpy(Anew.get(), A.get(), n * n * sizeof(double));

    
}

void deallocate(double *A, double *Anew, double* error_matrix)
{

    A = nullptr;
    Anew = nullptr;	
    error_matrix = nullptr;

}


__global__
void Calculate_matrix(double* A, double* Anew, size_t size)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i * size + j > size * size) return;
	
	if(!((j == 0 || i == 0 || j == size - 1 || i == size - 1)))
	{
		Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] +
							A[(i + 1) * size + j] + A[i * size + j + 1]);		
	}
}


__global__ 
void Error_matrix(double *Anew, double *A,double *error,int size)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i * size + j > size * size) return;

    error[i * size + j] = fabs(A[i * size + j] - Anew[i * size + j]);
}


int main(int argc, char const *argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")
        ("precision", po::value<double>()->default_value(0.000001), "Set precision")
        ("grid-size", po::value<int>()->default_value(33), "Set grid size")
        ("iterations", po::value<int>()->default_value(1000000), "Set number of iterations");

    po::variables_map vm;

    po::store(po::parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }
    po::notify(vm);


    double precision = vm["precision"].as<double>();
    int n = vm["grid-size"].as<int>();
    int iter_max= vm["iterations"].as<int>();
   

    std::unique_ptr<double[]> A_ptr(new double[n*n]);
    std::unique_ptr<double[]> Anew_ptr(new double[n*n]);
    initialize(std::ref(A_ptr), std::ref(Anew_ptr), n);
    std::unique_ptr<double[]> E_ptr(new double[n*n]);

	double* A = A_ptr.get();
    double* Anew = Anew_ptr.get();
    double* error_matrix = E_ptr.get();
    

    cuda_unique_ptr<cudaStream_t> stream_ptr(cuda_new_stream(),cuda_delete_stream);
    cuda_unique_ptr<cudaGraph_t>graph(cuda_new_graph(),cuda_delete_graph);
    cuda_unique_ptr<cudaGraphExec_t>graph_save(cuda_new_graph_save(),cuda_delete_graph_save);

    auto stream = *stream_ptr;
  
	cuda_unique_ptr<double> A_device_ptr(cuda_new<double>(sizeof(double)*n*n), cuda_delete<double>);
	cuda_unique_ptr<double> Anew_device_ptr(cuda_new<double>(sizeof(double)*n*n), cuda_delete<double>);
	cuda_unique_ptr<double> error_device_ptr(cuda_new<double>(sizeof(double)*n*n), cuda_delete<double>);
	cuda_unique_ptr<double> error_GPU_ptr(cuda_new<double>(sizeof(double)), cuda_delete<double>);


	double *error_GPU = error_GPU_ptr.get();
	double *error_device = error_device_ptr.get();
	double *Anew_device = Anew_device_ptr.get();
	double *A_device = A_device_ptr.get();

    cudaError_t cudaErr1 = cudaMemcpy(A_device,A,n*n*sizeof(double),cudaMemcpyHostToDevice);
    cudaError_t cudaErr2 = cudaMemcpy(Anew_device,Anew,n*n*sizeof(double),cudaMemcpyHostToDevice);

	if (cudaErr1 != cudaSuccess || cudaErr2 != cudaSuccess){
		std::cerr << "Memory transfering error" << std::endl;
		exit(3);
	}

	cuda_unique_ptr<double> tmp_ptr_old(cuda_new<double>(0), cuda_delete<double>);
	double *tmp_old = tmp_ptr_old.get();
    size_t tmp_size_old = 0;
    
    cub::DeviceReduce::Max(tmp_old,tmp_size_old,Anew_device,error_GPU,n*n);
	
	size_t tmp_size = tmp_size_old;
	cuda_unique_ptr<double> tmp_ptr(cuda_new<double>(tmp_size), cuda_delete<double>);
	double *tmp = tmp_ptr.get();


    int size_bl = 0;
    if(n % 32 == 0){
        size_bl = 32;
    }
    else{
        size_bl = n % 32;
    }

    dim3 block = dim3(size_bl, size_bl);
    dim3 grid((n + block.x - 1) /  block.x, (n + block.y - 1) /  block.y);


	int flag_graph = 0;
   
	double error = 1.0;
    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();

	while (error > precision && iter < iter_max){
		if(flag_graph == 1){
			cudaGraphLaunch(*graph_save, stream);
			cub::DeviceReduce::Max(tmp,tmp_size,error_device,error_GPU,n*n,stream);
			cudaMemcpy(&error,error_GPU,sizeof(double),cudaMemcpyDeviceToHost);

			iter += 100;
			printf("%5d, %0.6f\n", iter, error);
		}
        
		else{
			cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);

			for(size_t i =0 ; i < 99;i++){
				Calculate_matrix<<<grid, block, 0, stream>>>(Anew_device, A_device, n);

				double* temp = A_device;
				A_device = Anew_device;
				Anew_device = temp;
			}

			Calculate_matrix<<<grid, block, 0, stream>>>(Anew_device,A_device,n);
			Error_matrix<<<grid, block, 0, stream>>>(Anew_device,A_device,error_device,n);

			cudaStreamEndCapture(stream, graph.get());
			cudaGraphInstantiate(graph_save.get(), *graph, NULL, NULL, 0);

			flag_graph = 1;
		}

    }

	auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - start;
    
    printf("%5d, %0.6f\n", iter, error);
	printf(" total: %f s\n", runtime);

#ifdef TEST
    cudaErr1 = cudaMemcpy(A,A_device,sizeof(double)*n*n,cudaMemcpyDeviceToHost);

	if (cudaErr1 != cudaSuccess){
		std::cerr << "Memory transfering error" << std::endl;
		exit(3);
	}
	
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%0.6f ", A[i*n+j]); 
        }
        printf("\n");
    }
#endif


    return 0;
}
