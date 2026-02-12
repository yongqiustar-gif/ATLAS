#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t _err = (stmt);                                                \
        if (_err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_err));                               \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

__global__ void touch_kernel(int* x) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(x, 1);
    }
}

static double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}

int main(int argc, char** argv) {
    int iters = 200000;
    int warmup = 20000;
    int chunk = 1000;
    int sync_iters = 50000;
    if (argc > 1) iters = std::atoi(argv[1]);
    if (argc > 2) warmup = std::atoi(argv[2]);
    if (argc > 3) chunk = std::atoi(argv[3]);
    if (argc > 4) sync_iters = std::atoi(argv[4]);

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    int* d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));

    // Calibrate timer overhead (two chrono calls)
    {
        std::vector<double> t_over;
        t_over.reserve(100000);
        for (int i = 0; i < 100000; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto t1 = std::chrono::high_resolution_clock::now();
            t_over.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        double mean = std::accumulate(t_over.begin(), t_over.end(), 0.0) / t_over.size();
        std::printf("timer_overhead_us_mean=%.6f\n", mean);
    }

    // Method A: async enqueue timing around launch API only
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    for (int i = 0; i < warmup; ++i) {
        touch_kernel<<<1, 1>>>(d_counter);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> launch_us;
    launch_us.reserve(iters);
    auto t0_all = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        touch_kernel<<<1, 1>>>(d_counter);
        CUDA_CHECK(cudaGetLastError());
        auto t1 = std::chrono::high_resolution_clock::now();
        launch_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        if (chunk > 0 && (i + 1) % chunk == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1_all = std::chrono::high_resolution_clock::now();

    int counter_async = 0;
    CUDA_CHECK(cudaMemcpy(&counter_async, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    double launch_mean = std::accumulate(launch_us.begin(), launch_us.end(), 0.0) / launch_us.size();
    double launch_p50 = percentile(launch_us, 0.50);
    double launch_p90 = percentile(launch_us, 0.90);
    double launch_p99 = percentile(launch_us, 0.99);
    double wall_us_per_launch = std::chrono::duration<double, std::micro>(t1_all - t0_all).count() / iters;

    std::printf("methodA_async_enqueue_us_mean=%.6f\n", launch_mean);
    std::printf("methodA_async_enqueue_us_p50=%.6f\n", launch_p50);
    std::printf("methodA_async_enqueue_us_p90=%.6f\n", launch_p90);
    std::printf("methodA_async_enqueue_us_p99=%.6f\n", launch_p99);
    std::printf("methodA_wall_us_per_launch=%.6f\n", wall_us_per_launch);
    std::printf("methodA_counter=%d expected=%d\n", counter_async, warmup + iters);

    // Method B: launch + stream sync per iteration (end-to-end step)
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    for (int i = 0; i < warmup / 10; ++i) {
        touch_kernel<<<1, 1>>>(d_counter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> sync_step_us;
    sync_step_us.reserve(sync_iters);
    for (int i = 0; i < sync_iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        touch_kernel<<<1, 1>>>(d_counter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        sync_step_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    int counter_sync = 0;
    CUDA_CHECK(cudaMemcpy(&counter_sync, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    double sync_mean = std::accumulate(sync_step_us.begin(), sync_step_us.end(), 0.0) / sync_step_us.size();
    double sync_p50 = percentile(sync_step_us, 0.50);
    double sync_p90 = percentile(sync_step_us, 0.90);
    double sync_p99 = percentile(sync_step_us, 0.99);

    std::printf("methodB_launch_sync_us_mean=%.6f\n", sync_mean);
    std::printf("methodB_launch_sync_us_p50=%.6f\n", sync_p50);
    std::printf("methodB_launch_sync_us_p90=%.6f\n", sync_p90);
    std::printf("methodB_launch_sync_us_p99=%.6f\n", sync_p99);
    std::printf("methodB_counter=%d expected=%d\n", counter_sync, warmup / 10 + sync_iters);

    // Method C: GPU event envelope (kernel execution time only, not host launch)
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
        touch_kernel<<<1, 1>>>(d_counter);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));
    int counter_event = 0;
    CUDA_CHECK(cudaMemcpy(&counter_event, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    std::printf("methodC_gpu_event_us_per_kernel=%.6f\n", (gpu_ms * 1000.0) / iters);
    std::printf("methodC_counter=%d expected=%d\n", counter_event, iters);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    CUDA_CHECK(cudaFree(d_counter));
    return 0;
}
