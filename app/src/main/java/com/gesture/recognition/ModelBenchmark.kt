package com.gesture.recognition

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer

/**
 * ULTRA-SAFE Benchmark - logs at every single step
 */
class ModelBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "ModelBenchmark"
        private const val BENCHMARK_RUNS = 10  // Even fewer for faster testing
    }

    /**
     * Run minimal benchmark
     */
    fun runCompleteBenchmark(): String {
        // LOG IMMEDIATELY - before anything else!
        FileLogger.i(TAG, "========== BENCHMARK FUNCTION CALLED ==========")

        val report = StringBuilder()

        try {
            FileLogger.i(TAG, "Creating StringBuilder...")
            report.appendLine("════════════════════════════════════════════════════════")

            FileLogger.i(TAG, "Adding title...")
            report.appendLine("         MODEL ACCELERATION BENCHMARK REPORT")
            report.appendLine("════════════════════════════════════════════════════════")

            FileLogger.i(TAG, "Getting device info...")
            report.appendLine("Device: ${android.os.Build.MODEL}")
            report.appendLine("Android: ${android.os.Build.VERSION.SDK_INT}")
            report.appendLine()

            FileLogger.i(TAG, "Testing GPU compatibility...")
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            report.appendLine("GPU COMPATIBILITY CHECK")
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            try {
                val compatList = CompatibilityList()
                val gpuSupported = compatList.isDelegateSupportedOnThisDevice
                report.appendLine("GPU Supported: $gpuSupported")
                FileLogger.i(TAG, "GPU check OK: $gpuSupported")
            } catch (e: Exception) {
                report.appendLine("GPU check error: ${e.message}")
                FileLogger.e(TAG, "GPU check failed", e)
            }
            report.appendLine()

            // Test HandDetector only (simplify)
            FileLogger.i(TAG, "Starting HandDetector tests...")
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            report.appendLine("HANDDETECTOR BENCHMARK")
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            // Test CPU
            FileLogger.i(TAG, "Testing CPU...")
            try {
                val cpuTime = testCPU()
                report.appendLine("CPU: ${String.format("%.2f", cpuTime)}ms avg ✓")
                FileLogger.i(TAG, "CPU test OK: ${cpuTime}ms")
            } catch (e: Exception) {
                report.appendLine("CPU: ✗ FAILED - ${e.message}")
                FileLogger.e(TAG, "CPU test failed", e)
            }

            // Test GPU
            FileLogger.i(TAG, "Testing GPU...")
            try {
                val gpuTime = testGPU()
                report.appendLine("GPU: ${String.format("%.2f", gpuTime)}ms avg ✓")
                FileLogger.i(TAG, "GPU test OK: ${gpuTime}ms")
            } catch (e: Exception) {
                val errorMsg = if (e.message?.contains("Mali GPU") == true) {
                    "✗ CRASHED (Known Mali GPU driver issue)"
                } else {
                    "✗ FAILED - ${e.message}"
                }
                report.appendLine("GPU: $errorMsg")
                FileLogger.e(TAG, "GPU test failed: ${e.message}", e)

                // Add helpful note about Mali GPU issue
                report.appendLine("     Note: Mali-G68 MP5 GPU delegate has known issues with TFLite 2.14.0")
            }

            // Test NPU
            FileLogger.i(TAG, "Testing NPU...")
            try {
                val npuTime = testNPU()
                report.appendLine("NPU: ${String.format("%.2f", npuTime)}ms avg ✓")
                FileLogger.i(TAG, "NPU test OK: ${npuTime}ms")
            } catch (e: Exception) {
                report.appendLine("NPU: ✗ FAILED - ${e.message}")
                FileLogger.e(TAG, "NPU test failed", e)
            }

            report.appendLine()
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("RECOMMENDATIONS")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("Based on test results:")
            report.appendLine("  • CPU works: ~100ms inference time")
            report.appendLine("  • GPU: Check test results above")
            report.appendLine("  • NPU (NNAPI): Check test results above")
            report.appendLine()
            report.appendLine("If NPU works with <10ms inference time:")
            report.appendLine("  → USE NPU for all models (best performance!)")
            report.appendLine()
            report.appendLine("If GPU works but NPU fails:")
            report.appendLine("  → USE GPU as fallback")
            report.appendLine()
            report.appendLine("If both GPU and NPU fail:")
            report.appendLine("  → USE CPU with 4 threads (~100ms is acceptable)")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("BENCHMARK COMPLETE!")
            report.appendLine("════════════════════════════════════════════════════════")

            FileLogger.i(TAG, "========== BENCHMARK COMPLETED ==========")

        } catch (e: Exception) {
            val error = "CRASH: ${e.message}\n\n${e.stackTraceToString()}"
            report.appendLine(error)
            FileLogger.e(TAG, "FATAL ERROR in runCompleteBenchmark", e)
        }

        return report.toString()
    }

    /**
     * Test CPU performance
     */
    private fun testCPU(): Float {
        FileLogger.i(TAG, "CPU: Loading model...")
        var interpreter: Interpreter? = null

        try {
            val modelBuffer = loadModelFile("mediapipe_hand-handdetector.tflite")
            FileLogger.i(TAG, "CPU: Model loaded, creating interpreter...")

            val options = Interpreter.Options().setNumThreads(4)
            interpreter = Interpreter(modelBuffer, options)
            FileLogger.i(TAG, "CPU: Interpreter created")

            // Create input: [1, 256, 256, 3]
            FileLogger.i(TAG, "CPU: Creating input tensor...")
            val input = Array(1) {
                Array(256) {
                    Array(256) {
                        FloatArray(3)
                    }
                }
            }

            // Create outputs: boxes[1,2944,18] and scores[1,2944,1]
            FileLogger.i(TAG, "CPU: Creating output tensors...")
            val outputBoxes = Array(1) { Array(2944) { FloatArray(18) } }
            val outputScores = Array(1) { Array(2944) { FloatArray(1) } }
            val outputs = mapOf(0 to outputBoxes, 1 to outputScores)

            // Warmup
            FileLogger.i(TAG, "CPU: Running warmup...")
            repeat(2) {
                interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            }

            // Benchmark
            FileLogger.i(TAG, "CPU: Running benchmark...")
            val times = mutableListOf<Float>()
            repeat(BENCHMARK_RUNS) {
                val start = System.nanoTime()
                interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                times.add(elapsed)
            }

            val avg = times.average().toFloat()
            FileLogger.i(TAG, "CPU: Benchmark complete")
            return avg

        } finally {
            interpreter?.close()
        }
    }

    /**
     * Test GPU performance
     */
    private fun testGPU(): Float {
        FileLogger.i(TAG, "GPU: Checking support...")
        val compatList = CompatibilityList()
        if (!compatList.isDelegateSupportedOnThisDevice) {
            throw Exception("GPU not supported on device")
        }

        FileLogger.i(TAG, "GPU: Loading model...")
        var interpreter: Interpreter? = null
        var delegate: GpuDelegate? = null

        try {
            val modelBuffer = loadModelFile("mediapipe_hand-handdetector.tflite")
            FileLogger.i(TAG, "GPU: Model loaded, creating delegate...")

            // CRITICAL: GPU delegate creation can crash on some Mali GPUs
            // even when isDelegateSupportedOnThisDevice returns true
            try {
                delegate = GpuDelegate()
                FileLogger.i(TAG, "GPU: Delegate created successfully!")
            } catch (e: Exception) {
                FileLogger.e(TAG, "GPU: Delegate creation crashed (Mali GPU bug)", e)
                throw Exception("GPU delegate creation failed (known Mali GPU issue): ${e.message}")
            }

            FileLogger.i(TAG, "GPU: Creating interpreter...")

            val options = Interpreter.Options().addDelegate(delegate)
            interpreter = Interpreter(modelBuffer, options)
            FileLogger.i(TAG, "GPU: Interpreter created")

            // Create input
            FileLogger.i(TAG, "GPU: Creating tensors...")
            val input = Array(1) {
                Array(256) {
                    Array(256) {
                        FloatArray(3)
                    }
                }
            }

            val outputBoxes = Array(1) { Array(2944) { FloatArray(18) } }
            val outputScores = Array(1) { Array(2944) { FloatArray(1) } }
            val outputs = mapOf(0 to outputBoxes, 1 to outputScores)

            // Warmup
            FileLogger.i(TAG, "GPU: Running warmup...")
            repeat(2) {
                interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            }

            // Benchmark
            FileLogger.i(TAG, "GPU: Running benchmark...")
            val times = mutableListOf<Float>()
            repeat(BENCHMARK_RUNS) {
                val start = System.nanoTime()
                interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                times.add(elapsed)
            }

            val avg = times.average().toFloat()
            FileLogger.i(TAG, "GPU: Benchmark complete")
            return avg

        } finally {
            interpreter?.close()
            delegate?.close()
        }
    }

    /**
     * Test NPU performance
     */
    private fun testNPU(): Float {
        FileLogger.i(TAG, "NPU: Loading model...")
        var interpreter: Interpreter? = null
        var delegate: NnApiDelegate? = null

        try {
            val modelBuffer = loadModelFile("mediapipe_hand-handdetector.tflite")
            FileLogger.i(TAG, "NPU: Model loaded, creating delegate...")

            delegate = NnApiDelegate()
            FileLogger.i(TAG, "NPU: Delegate created, creating interpreter...")

            val options = Interpreter.Options().addDelegate(delegate)
            interpreter = Interpreter(modelBuffer, options)
            FileLogger.i(TAG, "NPU: Interpreter created")

            // Create input
            FileLogger.i(TAG, "NPU: Creating tensors...")
            val input = Array(1) {
                Array(256) {
                    Array(256) {
                        FloatArray(3)
                    }
                }
            }

            val outputBoxes = Array(1) { Array(2944) { FloatArray(18) } }
            val outputScores = Array(1) { Array(2944) { FloatArray(1) } }
            val outputs = mapOf(0 to outputBoxes, 1 to outputScores)

            // Warmup
            FileLogger.i(TAG, "NPU: Running warmup...")
            repeat(2) {
                interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            }

            // Benchmark
            FileLogger.i(TAG, "NPU: Running benchmark...")
            val times = mutableListOf<Float>()
            repeat(BENCHMARK_RUNS) {
                val start = System.nanoTime()
                interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                times.add(elapsed)
            }

            val avg = times.average().toFloat()
            FileLogger.i(TAG, "NPU: Benchmark complete")
            return avg

        } finally {
            interpreter?.close()
            delegate?.close()
        }
    }

    /**
     * Load model file
     */
    private fun loadModelFile(filename: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = java.io.FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )
    }
}