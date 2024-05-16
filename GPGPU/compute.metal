#include <metal_stdlib>
using namespace metal;

kernel void addition_compute_function(constant float* arr1 [[ buffer(0) ]],
									  constant float* arr2 [[ buffer(1) ]],
									  device float* arr3 [[ buffer(2) ]],
									  uint index [[ thread_position_in_grid ]]){
	arr3[index] = arr1[index] + arr2[index];
}

kernel void reduce_function(constant float* arr1 [[ buffer(0) ]],
							device atomic_float* s [[ buffer(1) ]],
							uint index [[ thread_position_in_grid ]]){
	atomic_fetch_add_explicit(s, arr1[index], memory_order_relaxed);
}
