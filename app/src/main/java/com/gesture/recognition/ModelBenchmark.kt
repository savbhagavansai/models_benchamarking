package com.gesture.recognition

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer

/**
 * MINIMAL Benchmark with extensive logging to find crash point
 */
class ModelBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "ModelBenchmark"
        private const val BENCHMARK_RUNS = 20  // Reduced for faster testing
    }

    data class ModelInfo(
        val name: String,
        val filename: String,
        val inputShape: IntArray
    )

    /**
     * Run minimal benchmark with extensive logging
     */
    fun runCompleteBenchmark(): String {
        val report = StringBuilder()

        try {
            FileLogger.i(TAG, "Step 1: Starting benchmark report...")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("         MODEL ACCELERATION BENCHMARK REPORT")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("Device: ${android.os.Build.MODEL}")
            report.appendLine("Android: ${android.os.Build.VERSION.SDK_INT}")
            report.appendLine()

            // Define models
            FileLogger.i(TAG, "Step 2: Defining models...")
            val models = listOf(
                ModelInfo("HandDetector", "mediapipe_hand-handdetector.tflite", intArrayOf(1, 256, 256, 3)),
                ModelInfo("HandLandmark", "mediapipe_hand-handlandmarkdetector.tflite", intArrayOf(1, 256, 256, 3))
            )
            FileLogger.i(TAG, "Models defined: ${models.size} models")

            // Test GPU device support
            FileLogger.i(TAG, "Step 3: Checking GPU support...")
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            report.appendLine("TEST 1: GPU DEVICE COMPATIBILITY")
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            try {
                val compatList = CompatibilityList()
                val gpuSupported = compatList.isDelegateSupportedOnThisDevice
                report.appendLine("GPU Delegate Supported: $gpuSupported")
                FileLogger.i(TAG, "GPU support check complete: $gpuSupported")
            } catch (e: Exception) {
                report.appendLine("GPU check failed: ${e.message}")
                FileLogger.e(TAG, "GPU check error: ${e.message}", e)
            }
            report.appendLine()

            // Benchmark each model
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            report.appendLine("TEST 2: PERFORMANCE BENCHMARK")
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            for (modelInfo in models) {
                FileLogger.i(TAG, "Testing model: ${modelInfo.name}")
                report.appendLine("\n▼ ${modelInfo.name}")
                report.appendLine("─".repeat(60))

                // Test CPU
                FileLogger.i(TAG, "${modelInfo.name}: Testing CPU...")
                try {
                    val cpuResult = benchmarkCPU(modelInfo)
                    report.appendLine(cpuResult)
                    FileLogger.i(TAG, "${modelInfo.name}: CPU test complete")
                } catch (e: Exception) {
                    report.appendLine("  CPU: ✗ FAILED - ${e.message}")
                    FileLogger.e(TAG, "${modelInfo.name}: CPU failed: ${e.message}", e)
                }

                // Test GPU
                FileLogger.i(TAG, "${modelInfo.name}: Testing GPU...")
                try {
                    val gpuResult = benchmarkGPU(modelInfo)
                    report.appendLine(gpuResult)
                    FileLogger.i(TAG, "${modelInfo.name}: GPU test complete")
                } catch (e: Exception) {
                    report.appendLine("  GPU: ✗ FAILED - ${e.message}")
                    FileLogger.e(TAG, "${modelInfo.name}: GPU failed: ${e.message}", e)
                }

                // Test NPU
                FileLogger.i(TAG, "${modelInfo.name}: Testing NPU...")
                try {
                    val npuResult = benchmarkNPU(modelInfo)
                    report.appendLine(npuResult)
                    FileLogger.i(TAG, "${modelInfo.name}: NPU test complete")
                } catch (e: Exception) {
                    report.appendLine("  NPU: ✗ FAILED - ${e.message}")
                    FileLogger.e(TAG, "${modelInfo.name}: NPU failed: ${e.message}", e)
                }
            }

            report.appendLine("\n════════════════════════════════════════════════════════")
            report.appendLine("BENCHMARK COMPLETE!")
            report.appendLine("════════════════════════════════════════════════════════")

            FileLogger.i(TAG, "Benchmark completed successfully!")

        } catch (e: Exception) {
            val error = "FATAL ERROR: ${e.message}\n\n${e.stackTraceToString()}"
            report.appendLine(error)
            FileLogger.e(TAG, "Fatal benchmark error: ${e.message}", e)
        }

        return report.toString()
    }

    /**
     * Benchmark with CPU
     */
    private fun benchmarkCPU(modelInfo: ModelInfo): String {
        var interpreter: Interpreter? = null

        try {
            FileLogger.i(TAG, "CPU: Loading model ${modelInfo.filename}...")
            val modelBuffer = loadModelFile(modelInfo.filename)

            FileLogger.i(TAG, "CPU: Creating interpreter...")
            val options = Interpreter.Options().setNumThreads(4)
            interpreter = Interpreter(modelBuffer, options)

            FileLogger.i(TAG, "CPU: Creating dummy input...")
            val input = createDummyInput(modelInfo.inputShape)

            FileLogger.i(TAG, "CPU: Running warmup...")
            repeat(3) {
                interpreter.run(input, input) // Simple run
            }

            FileLogger.i(TAG, "CPU: Running benchmark...")
            val times = mutableListOf<Float>()
            repeat(BENCHMARK_RUNS) {
                val start = System.nanoTime()
                interpreter.run(input, input)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                times.add(elapsed)
            }

            val avg = times.average().toFloat()
            FileLogger.i(TAG, "CPU: Benchmark complete - avg ${avg}ms")

            return "  CPU: ${String.format("%.2f", avg)}ms avg ✓"

        } finally {
            interpreter?.close()
        }
    }

    /**
     * Benchmark with GPU
     */
    private fun benchmarkGPU(modelInfo: ModelInfo): String {
        var interpreter: Interpreter? = null
        var delegate: GpuDelegate? = null

        try {
            FileLogger.i(TAG, "GPU: Checking compatibility...")
            val compatList = CompatibilityList()
            if (!compatList.isDelegateSupportedOnThisDevice) {
                FileLogger.w(TAG, "GPU: Not supported on device")
                return "  GPU: ✗ Not supported on this device"
            }

            FileLogger.i(TAG, "GPU: Loading model ${modelInfo.filename}...")
            val modelBuffer = loadModelFile(modelInfo.filename)

            FileLogger.i(TAG, "GPU: Creating GPU delegate...")
            delegate = GpuDelegate()

            FileLogger.i(TAG, "GPU: Creating interpreter with GPU delegate...")
            val options = Interpreter.Options().addDelegate(delegate)
            interpreter = Interpreter(modelBuffer, options)

            FileLogger.i(TAG, "GPU: Creating dummy input...")
            val input = createDummyInput(modelInfo.inputShape)

            FileLogger.i(TAG, "GPU: Running warmup...")
            repeat(3) {
                interpreter.run(input, input)
            }

            FileLogger.i(TAG, "GPU: Running benchmark...")
            val times = mutableListOf<Float>()
            repeat(BENCHMARK_RUNS) {
                val start = System.nanoTime()
                interpreter.run(input, input)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                times.add(elapsed)
            }

            val avg = times.average().toFloat()
            FileLogger.i(TAG, "GPU: Benchmark complete - avg ${avg}ms")

            return "  GPU: ${String.format("%.2f", avg)}ms avg ✓"

        } finally {
            interpreter?.close()
            delegate?.close()
        }
    }

    /**
     * Benchmark with NPU (NNAPI)
     */
    private fun benchmarkNPU(modelInfo: ModelInfo): String {
        var interpreter: Interpreter? = null
        var delegate: NnApiDelegate? = null

        try {
            FileLogger.i(TAG, "NPU: Loading model ${modelInfo.filename}...")
            val modelBuffer = loadModelFile(modelInfo.filename)

            FileLogger.i(TAG, "NPU: Creating NNAPI delegate...")
            delegate = NnApiDelegate()

            FileLogger.i(TAG, "NPU: Creating interpreter with NNAPI delegate...")
            val options = Interpreter.Options().addDelegate(delegate)
            interpreter = Interpreter(modelBuffer, options)

            FileLogger.i(TAG, "NPU: Creating dummy input...")
            val input = createDummyInput(modelInfo.inputShape)

            FileLogger.i(TAG, "NPU: Running warmup...")
            repeat(3) {
                interpreter.run(input, input)
            }

            FileLogger.i(TAG, "NPU: Running benchmark...")
            val times = mutableListOf<Float>()
            repeat(BENCHMARK_RUNS) {
                val start = System.nanoTime()
                interpreter.run(input, input)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                times.add(elapsed)
            }

            val avg = times.average().toFloat()
            FileLogger.i(TAG, "NPU: Benchmark complete - avg ${avg}ms")

            return "  NPU (NNAPI): ${String.format("%.2f", avg)}ms avg ✓"

        } finally {
            interpreter?.close()
            delegate?.close()
        }
    }

    /**
     * Load model file from assets
     */
    private fun loadModelFile(filename: String): ByteBuffer {
        FileLogger.i(TAG, "Loading model file: $filename")
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = java.io.FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val buffer = fileChannel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )
        FileLogger.i(TAG, "Model file loaded: ${buffer.capacity() / 1024} KB")
        return buffer
    }

    /**
     * Create dummy input tensor
     */
    private fun createDummyInput(shape: IntArray): Any {
        FileLogger.i(TAG, "Creating dummy input: shape=${shape.joinToString("x")}")
        return when (shape.size) {
            4 -> Array(shape[0]) {
                Array(shape[1]) {
                    Array(shape[2]) {
                        FloatArray(shape[3])
                    }
                }
            }
            else -> FloatArray(shape.reduce { acc, i -> acc * i })
        }
    }
}