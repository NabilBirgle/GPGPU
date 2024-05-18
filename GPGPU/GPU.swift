import MetalKit

class GPU {
	let device: MTLDevice?
	let commandQueue: MTLCommandQueue?
	let library: MTLLibrary?
	init() {
		self.device = MTLCreateSystemDefaultDevice()
		self.commandQueue = device?.makeCommandQueue()
		self.library = device?.makeDefaultLibrary()
	}
	var functions: [String: MTLFunction] = [:]
	func compile(name: String){
		functions[name] = library?.makeFunction(name: name)
	}
	var pipeline_state: MTLComputePipelineState?
	func set_gpgpu_pipeline(function_name: String){
		guard
			let f: MTLFunction = functions[function_name]
		else {
			return
		}
		do {
			pipeline_state = try device?.makeComputePipelineState(function: f)
		} catch {
			print(error)
		}
	}
}

extension GPU {
	class GPGPU{
		let gpu: GPU
		init(gpu: GPU) {
			self.gpu = gpu
		}
		var command_buffer: MTLCommandBuffer?
		var command_encoder: MTLComputeCommandEncoder?
		func call_kernel_function(){
			command_buffer = gpu.commandQueue?.makeCommandBuffer()
			command_encoder = command_buffer?.makeComputeCommandEncoder()
			command_encoder?.setComputePipelineState(gpu.pipeline_state!)
		}
		var input = 0
		func set_input(arr: [Float], n: Int){
			let buffer = gpu.device?.makeBuffer(bytes: arr,
												length: MemoryLayout<Float>.size * n,
												options: .storageModeShared)
			command_encoder?.setBuffer(buffer, offset: 0, index: input)
			input += 1
		}
		func set_input(arr: UnsafeMutablePointer<Float>, n: Int){
			let buffer = gpu.device?.makeBuffer(bytes: arr,
												length: MemoryLayout<Float>.size * n,
												options: .storageModeShared)
			command_encoder?.setBuffer(buffer, offset: 0, index: input)
			input += 1
		}
		func set_output(n: Int) -> UnsafeMutablePointer<Float> {
			let buffer = gpu.device?.makeBuffer(length: MemoryLayout<Float>.size * n,
												options: .storageModeShared)
			command_encoder?.setBuffer(buffer, offset: 0, index: input)
			input += 1
			return (buffer?.contents().bindMemory(
				to: Float.self, capacity: MemoryLayout<Float>.size * n))!
		}
		func set_index_input(thread: Int){
			let grid: MTLSize = MTLSize(width: thread, height: 1, depth: 1)
			let k: Int = gpu.pipeline_state!.maxTotalThreadsPerThreadgroup
			let subgrid: MTLSize = MTLSize(width: k, height: 1, depth: 1)
			command_encoder?.dispatchThreads(grid, threadsPerThreadgroup: subgrid)
			command_encoder?.endEncoding()
			command_buffer?.commit()
			command_buffer?.waitUntilCompleted()
			input = 0
		}
	}
}
