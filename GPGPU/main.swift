import MetalKit

let count: Int = 300000

// Create our random arrays
var array1 = getRandomArray()
var array2 = getRandomArray()

// Call our functions
computeWay(arr1: array1, arr2: array2)
basicForLoopWay(arr1: array1, arr2: array2)

func computeWay(arr1: [Float], arr2: [Float]) {
	// Begin the process
	let startTime = CFAbsoluteTimeGetCurrent()

	let gpu: GPU = GPU()
	let gpgpu: GPU.GPGPU = GPU.GPGPU(gpu: gpu)

	let f = "addition_compute_function"
	let g = "reduce_function"
	gpu.compile(name: f)
	gpu.compile(name: g)

	print()
	print("GPU Way")

	gpu.set_gpgpu_pipeline(function_name: f)
	gpgpu.call_kernel_function()
	gpgpu.set_input(arr: arr1, n: count)
	gpgpu.set_input(arr: arr2, n: count)
	let arr3 = gpgpu.set_output(n: count)
	gpgpu.set_index_input(thread: count)

	gpu.set_gpgpu_pipeline(function_name: g)
	gpgpu.call_kernel_function()
	gpgpu.set_input(arr: arr3, n: count)
	let arr4 = gpgpu.set_output(n: 1)
	gpgpu.set_index_input(thread: count)


	// Print out all of our new added together array information
	for i in 0..<3 {
		print("\(arr1[i]) + \(arr2[i]) = \(arr3[i])")
	}
	print("Sum(arr3) = \(arr4[0])")

	// Print out the elapsed time
	let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
	print("Time elapsed \(String(format: "%.05f", timeElapsed)) seconds")
	print()
}

func basicForLoopWay(arr1: [Float], arr2: [Float]) {
	print("CPU Way")

	// Begin the process
	let startTime = CFAbsoluteTimeGetCurrent()

	var arr3 = [Float].init(repeating: 0.0, count: count)

	// Process our additions of the arrays together
	for i in 0..<count {
		arr3[i] = arr1[i] + arr2[i]
	}

	// Print out the results
	for i in 0..<3 {
		print("\(arr1[i]) + \(arr2[i]) = \(arr3[i])")
	}
	let s: Float = arr3.reduce(0, +)
	print("Sum(arr3) = \(s)")

	// Print out the elapsed time
	let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
	print("Time elapsed \(String(format: "%.05f", timeElapsed)) seconds")
	print()
}

// Helper function
func getRandomArray() -> [Float] {
	var result = [Float].init(repeating: 0.0, count: count)
	for i in 0..<count {
		result[i] = Float(arc4random_uniform(10))
	}
	return result
}
