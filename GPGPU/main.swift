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
	let command_queue = Command_queue(gpu: gpu)

	let f = "addition_compute_function"
	let g = "reduce_function"
	gpu.compile(name: f)
	gpu.compile(name: g)

	print()
	print("GPU Way")

	var command_buffer: Command_buffer
	var command_encoder: Compute_command_encoder

	guard 
		let A1: MTLBuffer = gpu.get_device()?.makeBuffer(bytes: arr1,
											  length: MemoryLayout<Float>.size * count,
											  options: .storageModeShared),
		let A2: MTLBuffer = gpu.get_device()?.makeBuffer(bytes: arr2,
											  length: MemoryLayout<Float>.size * count,
											  options: .storageModeShared),
		let A3: MTLBuffer = gpu.get_device()?.makeBuffer(
											  length: MemoryLayout<Float>.size * count,
											  options: .storageModeShared),
		let A4: MTLBuffer = gpu.get_device()?.makeBuffer(
											  length: MemoryLayout<Float>.size,
											  options: .storageModeShared)
	else {
		return
	}

	command_buffer = Command_buffer(command_queue: command_queue)
	command_encoder = Compute_command_encoder(command_buffer: command_buffer)
	gpu.set_compute_pipeline_state(function_name: f)
	command_encoder.set_input(arr: A1)
	command_encoder.set_input(arr: A2)
	command_encoder.set_input(arr: A3)
	command_encoder.set_index_input(thread: count, compute_pipeline_state: gpu.get_compute_pipeline_state())
	command_encoder.end()
	command_buffer.commit()

	command_buffer = Command_buffer(command_queue: command_queue)
	command_encoder = Compute_command_encoder(command_buffer: command_buffer)
	gpu.set_compute_pipeline_state(function_name: g)
	command_encoder.set_input(arr: A3)
	command_encoder.set_input(arr: A4)
	command_encoder.set_index_input(thread: count, compute_pipeline_state: gpu.get_compute_pipeline_state())
	command_encoder.end()
	command_buffer.commit()

	let B3 = (A3.contents().bindMemory(
		to: Float.self, capacity: MemoryLayout<Float>.size * count))
	let B4 = (A4.contents().bindMemory(
		to: Float.self, capacity: MemoryLayout<Float>.size))

	// Print out all of our new added together array information
	for i in 0..<3 {
		print("\(arr1[i]) + \(arr2[i]) = \(B3[i])")
	}
	print("Sum(arr3) = \(B4[0])")

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
