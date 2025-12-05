#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>


//use multiple kernels to reduce since their is no all thread block sync, continue recurs
__global__ void interleaved(const float* input, float* output, int N) {
    //use shared memory (shared by only by threads (per block)), load global into thread specific mem
    extern __shared__  float s_input[];
    unsigned int index = threadIdx.x + blockDim.x * 2 * blockIdx.x; //2 thread for better util
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    s_input[threadIdx.x] = 0.0f;
    //this way of loading, loads in data from 2 blocks at a time, the curr block and the one ahead.
    while (index < N) {
        s_input[threadIdx.x] += input[index];
        if (index + blockDim.x < N) {
            s_input[threadIdx.x] += input[index + blockDim.x];
        }
        index += gridSize;
    }
    __syncthreads();

    

    //reduction (pow of 2 indexing)
    //Need to reverse bc otherwise memory would be overwritten if saved sequentially by threadidx
    for (unsigned int i = blockDim.x/2; i > 0; i >>= 1) { //>> 1 bit shift 1 is equivalent to dividing by 2
        //first iteration by 128, then by 64, then by 32...
        if (threadIdx.x < i) {
            s_input[threadIdx.x] += s_input[i + threadIdx.x]; //Halfway through iteration
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = s_input[0];
    }
    //Can unroll for further speed up
}



// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int threadsPerBlock, int blocksPerGrid) {  

    interleaved<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
    cudaDeviceSynchronize();

}

void generate_test_arrays(float** input_cases, float** output_cases, int* sizes, int cases, int threadsPerBlock) {
    for (int i = 0; i < cases; i++) {
        int N = rand() % 1000 + 100; // Smaller test cases for demonstration
        sizes[i] = N;
        
        input_cases[i] = new float[N];
        
        // Calculate output size based on blocks
        int blocksPerGrid = (N + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);
        output_cases[i] = new float[blocksPerGrid];
        
        // Generate random input data
        for (int j = 0; j < N; j++) {
            input_cases[i][j] = static_cast<float>(rand() % 100 + 1);
        }
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments or use defaults
    int threadsPerBlock = 256;  // Default value
    int blocksPerGrid = 0;      // Will be calculated if not provided
    
    if (argc > 1) {
        threadsPerBlock = atoi(argv[1]);
    }
    if (argc > 2) {
        blocksPerGrid = atoi(argv[2]);
    }
    
    std::cout << "Using threadsPerBlock: " << threadsPerBlock << std::endl;
    if (blocksPerGrid > 0) {
        std::cout << "Using fixed blocksPerGrid: " << blocksPerGrid << std::endl;
    } else {
        std::cout << "blocksPerGrid will be calculated based on input size" << std::endl;
    }
    
    int cases = 10;
    float** input_cases = new float*[cases];
    float** output_cases = new float*[cases];
    int* sizes = new int[cases];

    generate_test_arrays(input_cases, output_cases, sizes, cases, threadsPerBlock);

    for (int i = 0; i < cases; i++) {
        int N = sizes[i];
        int actualBlocksPerGrid = (blocksPerGrid > 0) ? blocksPerGrid : 
                                  (N + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);
        
        // Allocate device memory
        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, actualBlocksPerGrid * sizeof(float));
        
        // Copy input from host to device
        cudaMemcpy(d_input, input_cases[i], N * sizeof(float), cudaMemcpyHostToDevice);
        
        // Run the solve function with device pointers
        solve(d_input, d_output, N, threadsPerBlock, actualBlocksPerGrid);
        
        // Copy result back from device to host
        cudaMemcpy(output_cases[i], d_output, actualBlocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Print results (partial sums from each block)
        std::cout << "Case " << i << " (N=" << N << ", blocks=" << actualBlocksPerGrid << "): ";
        for (int j = 0; j < actualBlocksPerGrid; j++) {
            std::cout << output_cases[i][j] << " ";
        }
        std::cout << std::endl;
        
        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }

    // Clean up host memory
    for (int i = 0; i < cases; i++) {
        delete[] input_cases[i];
        delete[] output_cases[i];
    }
    delete[] input_cases;
    delete[] output_cases;
    delete[] sizes;

    return 0;
}