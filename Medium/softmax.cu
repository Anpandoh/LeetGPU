#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>




__global__ void softmax_kernel(const float* input, float* output, int N) {

    


     //use shared memory (shared by only by threads (per block)), load global into thread specific mem
     extern __shared__ float shared[];
     float* s_input = shared;
     float* m_input = shared + blockDim.x;

     unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
     
     s_input[threadIdx.x] = 0.0f;
     m_input[threadIdx.x] = -INFINITY;
     //load into 2 different shared arrays at once.
     //this way of loading, loads in data from 2 blocks at a time, the curr block and the one ahead.
     if (index < N) {
        m_input[threadIdx.x] = input[index];
     }

    __syncthreads(); //think of sync thread as a way to pause, makes sures the number is fully right
    
     //max trick, subtract the max value of input before exponention
     __shared__  float max;
     for (unsigned int i = blockDim.x/2; i > 0; i >>= 1) { //>> 1 bit shift 1 is equivalent to dividing by 2
        //first iteration by 128, then by 64, then by 32... and compares the max value from the other half
        //goes through all threads in the block
        if (threadIdx.x < i) {
            m_input[threadIdx.x] = fmaxf(m_input[threadIdx.x], m_input[i + threadIdx.x]); //Halfway through iteration
        }
        __syncthreads(); //think of sync thread as a way to pause, makes sures the number is fully right
    }
    if (threadIdx.x == 0) {
        max = m_input[0];
    }
    __syncthreads(); //think of sync thread as a way to pause, makes sures the number is fully right


    __shared__  float sum;
    if (index < N) {
        s_input[threadIdx.x] = expf(input[index] - max);
    }
    __syncthreads(); //think of sync thread as a way to pause, makes sures the number is fully right

     //softmax (pow of 2 indexing)
     //Need to reverse bc otherwise memory would be overwritten if saved sequentially by threadidx
     for (unsigned int i = blockDim.x/2; i > 0; i >>= 1) { //>> 1 bit shift 1 is equivalent to dividing by 2
         //first iteration by 128, then by 64, then by 32... and adds it to from the other half
         //goes through all threads in the block
         if (threadIdx.x < i) {
             s_input[threadIdx.x] += s_input[i + threadIdx.x]; //Halfway through iteration
         }
         __syncthreads();
     }
     
     if (threadIdx.x == 0) {
        sum = s_input[0];
     }
     __syncthreads(); //think of sync thread as a way to pause, makes sures the number is fully right


     if (index < N) {
        output[index] = expf(input[index]-max)/sum;
     }
     //Can unroll for further speed up

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(input, output, N);    cudaDeviceSynchronize();
}

void generate_test_arrays(float** input_cases, float** output_cases, int* sizes, int cases) {
    for (int i = 0; i < cases; i++) {
        int N = rand() % 1000 + 100;
        sizes[i] = N;
        
        input_cases[i] = new float[N];
        output_cases[i] = new float[N];
    }
}

int main(int argc, char* argv[]) {
    int cases = 10;
    float** input_cases = new float*[cases];
    float** output_cases = new float*[cases];
    int* sizes = new int[cases];

    generate_test_arrays(input_cases, output_cases, sizes, cases);
    for (int i = 0; i < cases; i++) {
        int N = sizes[i];
        float *d_input, *d_output;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));
        cudaMemcpy(d_input, input_cases[i], N * sizeof(float), cudaMemcpyHostToDevice);
        solve(d_input, d_output, N);
        cudaMemcpy(output_cases[i], d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    for (int i = 0; i < cases; i++) {
        int N = sizes[i];
        for (int j = 0; j < N; j++) {
            std::cout << output_cases[i][j] << " ";
        }
        std::cout << std::endl;
    }
    for (int i = 0; i < cases; i++) {
        delete[] input_cases[i];
        delete[] output_cases[i];
    }
    delete[] input_cases;
    delete[] output_cases;
    delete[] sizes;
    return 0;
}