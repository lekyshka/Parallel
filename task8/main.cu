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

#define OFFSET(x, y, m) (((x)*(m)) + (y))


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

    error[i * size + j] = fabs(A[i * size + j] - Anew[i * size + j]);
}


int main(int argc, char const *argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message")
        ("precision", po::value<double>()->default_value(0.000001), "Set precision")
        ("grid-size", po::value<int>()->default_value(256), "Set grid size")
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
    

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graph_save;


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

	cuda_unique_ptr<double> tmp_ptr(cuda_new<double>(0), cuda_delete<double>);
	double *tmp = tmp_ptr.get();
    size_t tmp_size = 0;
    
    cub::DeviceReduce::Max(tmp,tmp_size,Anew_device,error_GPU,n*n);


	cudaErr1 = cudaMalloc(&tmp,tmp_size);
	if (cudaErr1 != cudaSuccess){
		std::cerr << "Memory transfering error" << std::endl;
		exit(3);
	}

    dim3 block = dim3(32, 32);
    dim3 grid(n / block.x, n / block.y);

	int flag_graph = 0;
   
	double error = 1.0;
    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();

	while (error > precision && iter < iter_max){
		if(flag_graph == 1){
			cudaGraphLaunch(graph_save,stream);

			cudaMemcpyAsync(&error,error_GPU,sizeof(double),cudaMemcpyDeviceToHost, stream);
			cudaStreamSynchronize(stream);

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
			
			cub::DeviceReduce::Max(tmp,tmp_size,error_device,error_GPU,n*n,stream);

			cudaStreamEndCapture(stream, &graph);
			cudaGraphInstantiate(&graph_save, graph, NULL, NULL, 0);

			flag_graph = 1;
		}

    }

	auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - start;
    
    printf("%5d, %0.6f\n", iter, error);
	printf(" total: %f s\n", runtime);


    // cudaErr1 = cudaMemcpy(A,A_device,sizeof(double)*n*n,cudaMemcpyDeviceToHost);

	// if (cudaErr1 != cudaSuccess){
	// 	std::cerr << "Memory transfering error" << std::endl;
	// 	exit(3);
	// }
	
	// for (size_t i = 0; i < N; i++){
	// 	for (size_t j = 0; j < N; j++){

	// 		std::cout << A[i*N+j] << ' ';
			
	// 	}
    //     std::cout << std::endl;
    // }


    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
	deallocate(A, Anew, error_matrix);


    return 0;
}


// #include <iostream>
// #include <boost/program_options.hpp>
// #include <cmath>
// #include <memory>
// #include <algorithm>
// #include <fstream>
// #include <iomanip>
// #include <chrono>
// namespace opt = boost::program_options;

// #include <cuda_runtime.h>
// #include <cub/cub.cuh>


// #define CHECK(call)                                                             \
//     {                                                                           \
//         const cudaError_t error = call;                                         \
//         if (error != cudaSuccess)                                               \
//         {                                                                       \
//             printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
//             printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
//             exit(1);                                                            \
//         }                                                                       \
//     }

// // собственно возвращает значение линейной интерполяции
// double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
//     // делаем значение y(щначение клетки)используя формулу линейной интерполяции
//     return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
// }



// void initMatrix(std::unique_ptr<double[]> &arr ,int N){
        
//         for (size_t i = 0; i < N*N-1; i++)
//         {
//             arr[i] = 0;
//         }
        


//           arr[0] = 10.0;
//           arr[N-1] = 20.0;
//           arr[(N-1)*N + (N-1)] = 30.0;
//           arr[(N-1)*N] = 20.0;
//               // инициализируем и потом сразу отправим на девайс
//         for (size_t i = 1; i < N-1; i++)
//         {
//             arr[0*N+i] = linearInterpolation(i,0.0,arr[0],N-1,arr[N-1]);
//             arr[i*N+0] = linearInterpolation(i,0.0,arr[0],N-1,arr[(N-1)*N]);
//             arr[i*N+(N-1)] = linearInterpolation(i,0.0,arr[N-1],N-1,arr[(N-1)*N + (N-1)]);
//             arr[(N-1)*N+i] = linearInterpolation(i,0.0,arr[(N-1)*N],N-1,arr[(N-1)*N + (N-1)]);
//         }
// }




// void saveMatrixToFile(const double* matrix, int N, const std::string& filename) {
//     std::ofstream outputFile(filename);
//     if (!outputFile.is_open()) {
//         std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
//         return;
//     }

//     // Устанавливаем ширину вывода для каждого элемента
//     int fieldWidth = 10; // Ширина поля вывода, можно настроить по вашему усмотрению

//     // Записываем матрицу в файл с выравниванием столбцов
//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < N; ++j) {
//             outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * N + j];
//         }
//         outputFile << std::endl;
//     }

//     outputFile.close();
// }


// void swapMatrices(double* &prevmatrix, double* &curmatrix) {
//     double* temp = prevmatrix;
//     prevmatrix = curmatrix;
//     curmatrix = temp;
    
// }





// __global__ void computeOneIteration(double *prevmatrix, double *curmatrix, int size){
    
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     // чтобы не изменять границы
//     if (!(j == 0 || i == 0 || i >= size-1 || j >= size-1))
//         curmatrix[i*size+j]  = 0.25 * (prevmatrix[i*size+j+1] + prevmatrix[i*size+j-1] + prevmatrix[(i-1)*size+j] + prevmatrix[(i+1)*size+j]);
        

// }


// // вычитание из матрицы, результат сохраняем в матрицу пред. значений
// __global__ void matrixSub(double *prevmatrix, double *curmatrix,double *error,int size){
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;

//     // чтобы не изменять границы
//     // if (!(j == 0 || i == 0 || i >= size-1 || j >= size-1))
//         error[i*size + j] = fabs(curmatrix[i*size+j] - prevmatrix[i*size+j]);
        

// }


// int main(int argc, char const *argv[])
// {
//     // парсим аргументы
//     opt::options_description desc("опции");
//     desc.add_options()
//         ("accuracy",opt::value<double>()->default_value(1e-6),"точность")
//         ("cellsCount",opt::value<int>()->default_value(128),"размер матрицы")
//         ("iterCount",opt::value<int>()->default_value(1000000),"количество операций")
//         ("help","помощь")
//     ;

//     opt::variables_map vm;

//     opt::store(opt::parse_command_line(argc, argv, desc), vm);

//     opt::notify(vm);

//     if (vm.count("help")) {
//         std::cout << desc << "\n";
//         return 1;
//     }

    
//     // и это всё было только ради того чтобы спарсить аргументы.......

//     int N = vm["cellsCount"].as<int>();
//     double accuracy = vm["accuracy"].as<double>();
//     int countIter = vm["iterCount"].as<int>();
   
    
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);
//     cudaGraph_t     graph;
//     cudaGraphExec_t g_exec;

//     double *prevmatrix_GPU  = NULL;
//     double *error_GPU  = NULL;
//     // tmp будет буфером для хранения результатов редукции , по блокам и общий
//     double *tmp=NULL;
//     size_t tmp_size = 0;
//     double *curmatrix_GPU = NULL;

//     double error =1.0;
//     int iter = 0;

//     std::unique_ptr<double[]> A(new double[N*N]);
//     std::unique_ptr<double[]> Anew(new double[N*N]);
//     std::unique_ptr<double[]> B(new double[N*N]);

//     initMatrix(std::ref(A),N);
//     initMatrix(std::ref(Anew),N);
    
//     double* curmatrix = A.get();
//     double* prevmatrix = Anew.get();
//     double* error_matrix = B.get();
//     double* error_gpu;
//     CHECK(cudaMalloc(&curmatrix_GPU,sizeof(double)*N*N));
//     CHECK(cudaMalloc(&prevmatrix_GPU,sizeof(double)*N*N));
//     CHECK(cudaMalloc(&error_gpu,sizeof(double)*N*N));
//     CHECK(cudaMalloc(&error_GPU,sizeof(double)*1));
//     CHECK(cudaMemcpy(curmatrix_GPU,curmatrix,N*N*sizeof(double),cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(prevmatrix_GPU,prevmatrix,N*N*sizeof(double),cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(error_gpu,error_matrix,N*N*sizeof(double),cudaMemcpyHostToDevice));
    
    
//     cub::DeviceReduce::Max(tmp,tmp_size,prevmatrix_GPU,error_GPU,N*N);

//     CHECK(cudaMalloc(&tmp,tmp_size));



//     dim3 threads_in_block = dim3(32, 32);
//     dim3 blocks_in_grid((N + threads_in_block.x - 1) / threads_in_block.x, (N + threads_in_block.y - 1) / threads_in_block.y);



// // начало записи вычислительного графа
//     cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);
    
//         // 99 - считаем ошибку через 100 итераций

//     for(size_t i =0 ; i<99;i++){
        
//         // cudaDeviceSynchronize();
//         computeOneIteration<<<blocks_in_grid, threads_in_block,0,stream>>>(prevmatrix_GPU,curmatrix_GPU,N);
//         swapMatrices(prevmatrix_GPU,curmatrix_GPU);
//         // cudaDeviceSynchronize();
//         // cudaMemcpy(prevmatrix_GPU,curmatrix_GPU,N*N*sizeof(double),cudaMemcpyDeviceToDevice);

//     }

//     computeOneIteration<<<blocks_in_grid, threads_in_block,0,stream>>>(prevmatrix_GPU,curmatrix_GPU,N);
//     matrixSub<<<blocks_in_grid, threads_in_block,0,stream>>>(prevmatrix_GPU,curmatrix_GPU,error_gpu,N);
    


//     // cudaDeviceSynchronize();
//     cub::DeviceReduce::Max(tmp,tmp_size,error_gpu,error_GPU,N*N,stream);
//     cudaStreamEndCapture(stream, &graph);


//     // закончили построение выч. графа
    
    
//     // получили экземпляр выч.графа
//     cudaGraphInstantiate(&g_exec, graph, NULL, NULL, 0);

//     auto start = std::chrono::high_resolution_clock::now();
//     while(error > accuracy && iter < countIter){
//         cudaGraphLaunch(g_exec,stream);
//         // matrixSub<<<blocks_in_grid, threads_in_block,0,stream>>>(prevmatrix_GPU,curmatrix_GPU,error_gpu,N);
//         // cub::DeviceReduce::Max(tmp,tmp_size,error_gpu,error_GPU,N*N,stream);
//         // cudaDeviceSynchronize();
//         cudaMemcpy(&error,error_GPU,1*sizeof(double),cudaMemcpyDeviceToHost);
//         iter+=100;
//         std::cout << "iteration: "<<iter << ' ' <<"error: "<<error << std::endl;

//     }
    
    


//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration = end - start;
//     auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
    
//     std::cout<<"time: " << time_s<<" error: "<<error << " iterarion: " << iter<<std::endl;
    
//     CHECK(cudaMemcpy(prevmatrix,prevmatrix_GPU,sizeof(double)*N*N,cudaMemcpyDeviceToHost));
//     CHECK(cudaMemcpy(error_matrix,error_gpu,sizeof(double)*N*N,cudaMemcpyDeviceToHost));

//     CHECK(cudaMemcpy(curmatrix,curmatrix_GPU,sizeof(double)*N*N,cudaMemcpyDeviceToHost));
//     if (N <=13){
        
//         for (size_t i = 0; i < N; i++)
//         {
//             for (size_t j = 0; j < N; j++)
//             {
//                 /* code */
//                 std::cout << A[i*N+j] << ' ';
                
//             }
//             std::cout << std::endl;
//         }

//         for (size_t i = 0; i < N; i++)
//         {
//             for (size_t j = 0; j < N; j++)
//             {
//                 /* code */
//                 std::cout << Anew[i*N+j] << ' ';
                
//             }
//             std::cout << std::endl;
//         }

//     }
//     saveMatrixToFile(curmatrix, N , "matrix.txt");
//     cudaStreamDestroy(stream);
//     cudaGraphDestroy(graph);
//     cudaFree(prevmatrix_GPU);
//     cudaFree(curmatrix_GPU);
//     cudaFree(tmp);
//     cudaFree(error_GPU);
//     A = nullptr;
//     Anew = nullptr;

    
    

//     return 0;
// }
