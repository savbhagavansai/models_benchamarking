package com.gesture.recognition

import android.content.Context
import android.os.Build
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer

/**
 * COMPREHENSIVE BENCHMARK - Tests CPU, GPU, and NPU on all devices
 * FIXED: Uses simple GpuDelegate() API that works on TFLite 2.14.0
 */
class ModelBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "ModelBenchmark"
        private const val BENCHMARK_RUNS = 20
    }

    data class DeviceInfo(
        val manufacturer: String,
        val model: String,
        val hardware: String,
        val androidVersion: Int,
        val isQualcomm: Boolean,
        val isExynos: Boolean,
        val gpuName: String
    )

    /**
     * Run complete benchmark suite
     */
    fun runCompleteBenchmark(): String {
        FileLogger.i(TAG, "========== BENCHMARK STARTED ==========")

        val report = StringBuilder()

        try {
            // Detect device info
            val deviceInfo = detectDevice()

            FileLogger.i(TAG, "Building report...")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("      COMPREHENSIVE MODEL ACCELERATION BENCHMARK")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine()
            report.appendLine("DEVICE INFORMATION:")
            report.appendLine("  Manufacturer: ${deviceInfo.manufacturer}")
            report.appendLine("  Model: ${deviceInfo.model}")
            report.appendLine("  Hardware: ${deviceInfo.hardware}")
            report.appendLine("  Android Version: ${deviceInfo.androidVersion}")
            report.appendLine("  GPU: ${deviceInfo.gpuName}")
            report.appendLine()

            if (deviceInfo.isQualcomm) {
                report.appendLine("  ✓ Qualcomm Snapdragon detected")
                report.appendLine("  Expected: Excellent GPU/NPU support")
            } else if (deviceInfo.isExynos) {
                report.appendLine("  ⚠ Samsung Exynos detected")
                report.appendLine("  Warning: GPU delegate may crash (known issue)")
            } else {
                report.appendLine("  ℹ Unknown chipset")
                report.appendLine("  Note: GPU support may vary")
            }
            report.appendLine()

            // GPU Compatibility Check
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            report.appendLine("STEP 1: GPU COMPATIBILITY CHECK")
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            val gpuCompatInfo = checkGPUCompatibility()
            report.appendLine(gpuCompatInfo)

            // Performance Benchmark
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            report.appendLine("STEP 2: PERFORMANCE BENCHMARK (HandDetector Model)")
            report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            report.appendLine()

            var cpuTime = 0f
            var gpuTime = 0f
            var npuTime = 0f
            var cpuWorking = false
            var gpuWorking = false
            var npuWorking = false

            // Test CPU
            FileLogger.i(TAG, "Testing CPU...")
            try {
                cpuTime = testCPU()
                cpuWorking = true
                report.appendLine("✓ CPU: ${String.format("%.2f", cpuTime)}ms avg")
                report.appendLine("  Threads: 4")
                report.appendLine("  Status: WORKING")
                FileLogger.i(TAG, "CPU test OK: ${cpuTime}ms")
            } catch (e: Exception) {
                report.appendLine("✗ CPU: FAILED - ${e.message}")
                FileLogger.e(TAG, "CPU test failed", e)
            }
            report.appendLine()

            // Test GPU
            FileLogger.i(TAG, "Testing GPU...")
            try {
                gpuTime = testGPU()
                gpuWorking = true
                report.appendLine("✓ GPU: ${String.format("%.2f", gpuTime)}ms avg")
                report.appendLine("  Backend: ${deviceInfo.gpuName}")
                report.appendLine("  Status: WORKING ✓")
                if (cpuWorking) {
                    val speedup = cpuTime / gpuTime
                    report.appendLine("  Speedup vs CPU: ${String.format("%.1f", speedup)}x faster!")
                }
                FileLogger.i(TAG, "GPU test OK: ${gpuTime}ms")
            } catch (e: Exception) {
                report.appendLine("✗ GPU: FAILED")
                report.appendLine("  Error: ${e.message}")
                report.appendLine("  Status: NOT AVAILABLE")

                if (deviceInfo.isExynos) {
                    report.appendLine("  Note: This is a known issue with Samsung Exynos Mali GPUs")
                    report.appendLine("        TensorFlow Lite GPU delegate has driver incompatibilities")
                }

                FileLogger.e(TAG, "GPU test failed", e)
            }
            report.appendLine()

            // Test NPU (NNAPI)
            FileLogger.i(TAG, "Testing NPU (NNAPI)...")
            try {
                npuTime = testNPU()
                npuWorking = true

                // Check if it's actually NPU or CPU fallback
                val actuallyAccelerated = cpuWorking && (npuTime < cpuTime * 0.9f)

                if (actuallyAccelerated) {
                    report.appendLine("✓ NPU (NNAPI): ${String.format("%.2f", npuTime)}ms avg")
                    report.appendLine("  Backend: Hardware NPU/DSP")
                    report.appendLine("  Status: WORKING ✓")
                    if (cpuWorking) {
                        val speedup = cpuTime / npuTime
                        report.appendLine("  Speedup vs CPU: ${String.format("%.1f", speedup)}x faster!")
                    }
                } else {
                    report.appendLine("⚠ NPU (NNAPI): ${String.format("%.2f", npuTime)}ms avg")
                    report.appendLine("  Backend: CPU fallback (not accelerated)")
                    report.appendLine("  Status: Delegate loads but uses CPU")

                    if (deviceInfo.androidVersion >= 35) {
                        report.appendLine("  Note: NNAPI is deprecated on Android 15+")
                    }
                    if (deviceInfo.isExynos) {
                        report.appendLine("  Note: Samsung Exynos NPU not exposed via NNAPI")
                        report.appendLine("        Use Samsung Neural SDK for real NPU access")
                    }
                }

                FileLogger.i(TAG, "NPU test OK: ${npuTime}ms (accelerated: $actuallyAccelerated)")
            } catch (e: Exception) {
                report.appendLine("✗ NPU (NNAPI): FAILED")
                report.appendLine("  Error: ${e.message}")
                report.appendLine("  Status: NOT AVAILABLE")
                FileLogger.e(TAG, "NPU test failed", e)
            }

            report.appendLine()
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("RECOMMENDATIONS")
            report.appendLine("════════════════════════════════════════════════════════")

            val recommendations = generateRecommendations(
                deviceInfo, cpuWorking, gpuWorking, npuWorking,
                cpuTime, gpuTime, npuTime
            )
            report.appendLine(recommendations)

            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("BENCHMARK COMPLETE!")
            report.appendLine("════════════════════════════════════════════════════════")

            FileLogger.i(TAG, "========== BENCHMARK COMPLETED ==========")

        } catch (e: Exception) {
            val error = "FATAL ERROR: ${e.message}\n\n${e.stackTraceToString()}"
            report.appendLine(error)
            FileLogger.e(TAG, "Fatal benchmark error", e)
        }

        return report.toString()
    }

    /**
     * Detect device information
     */
    private fun detectDevice(): DeviceInfo {
        val manufacturer = Build.MANUFACTURER
        val model = Build.MODEL
        val hardware = Build.HARDWARE.lowercase()
        val androidVersion = Build.VERSION.SDK_INT

        // Detect Qualcomm
        val isQualcomm = hardware.contains("qcom") ||
                        hardware.contains("qualcomm") ||
                        hardware.contains("snapdragon")

        // Detect Samsung Exynos
        val isExynos = hardware.contains("exynos") ||
                      (manufacturer.equals("samsung", ignoreCase = true) && !isQualcomm)

        // Try to detect GPU name
        val gpuName = when {
            isQualcomm -> "Qualcomm Adreno"
            isExynos -> "ARM Mali"
            else -> "Unknown GPU"
        }

        FileLogger.i(TAG, "Device detected: $manufacturer $model ($hardware)")
        FileLogger.i(TAG, "Qualcomm: $isQualcomm, Exynos: $isExynos, Android: $androidVersion")

        return DeviceInfo(manufacturer, model, hardware, androidVersion, isQualcomm, isExynos, gpuName)
    }

    /**
     * Check GPU compatibility using official TFLite CompatibilityList
     */
    private fun checkGPUCompatibility(): String {
        val report = StringBuilder()

        try {
            FileLogger.i(TAG, "Checking GPU compatibility...")

            val compatList = CompatibilityList()
            val isSupported = compatList.isDelegateSupportedOnThisDevice

            report.appendLine("GPU Delegate Support: ${if (isSupported) "YES ✓" else "NO ✗"}")
            report.appendLine()

            if (isSupported) {
                report.appendLine("The device reports GPU delegate is supported.")
                report.appendLine("Will attempt GPU benchmark in next step.")
                FileLogger.i(TAG, "GPU compatibility check: SUPPORTED")
            } else {
                report.appendLine("The device reports GPU delegate is NOT supported.")
                report.appendLine("GPU benchmark will be skipped.")
                FileLogger.w(TAG, "GPU compatibility check: NOT SUPPORTED")
            }

        } catch (e: Exception) {
            report.appendLine("GPU compatibility check FAILED: ${e.message}")
            FileLogger.e(TAG, "GPU compatibility check error", e)
        }

        report.appendLine()
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
            FileLogger.i(TAG, "CPU: Creating interpreter...")

            val options = Interpreter.Options()
                .setNumThreads(4)

            interpreter = Interpreter(modelBuffer, options)
            FileLogger.i(TAG, "CPU: Interpreter created")

            return runBenchmark(interpreter, "CPU")

        } finally {
            interpreter?.close()
        }
    }

    /**
     * Test GPU performance - FIXED: Simple API for TFLite 2.14.0
     */
    private fun testGPU(): Float {
        FileLogger.i(TAG, "GPU: Checking compatibility...")
        var interpreter: Interpreter? = null
        var delegate: GpuDelegate? = null

        try {
            val compatList = CompatibilityList()

            if (!compatList.isDelegateSupportedOnThisDevice) {
                throw Exception("GPU not supported on this device")
            }

            FileLogger.i(TAG, "GPU: Loading model...")
            val modelBuffer = loadModelFile("mediapipe_hand-handdetector.tflite")

            FileLogger.i(TAG, "GPU: Creating GPU delegate...")

            // FIXED: Use simple GpuDelegate() constructor
            // This works on TFLite 2.14.0 without GpuDelegateFactory dependency
            delegate = GpuDelegate()

            FileLogger.i(TAG, "GPU: Delegate created successfully")

            FileLogger.i(TAG, "GPU: Creating interpreter...")
            val options = Interpreter.Options()
                .addDelegate(delegate)

            interpreter = Interpreter(modelBuffer, options)
            FileLogger.i(TAG, "GPU: Interpreter created with GPU delegate")

            return runBenchmark(interpreter, "GPU")

        } finally {
            interpreter?.close()
            delegate?.close()
        }
    }

    /**
     * Test NPU (NNAPI) performance
     */
    private fun testNPU(): Float {
        FileLogger.i(TAG, "NPU: Loading model...")
        var interpreter: Interpreter? = null
        var delegate: NnApiDelegate? = null

        try {
            val modelBuffer = loadModelFile("mediapipe_hand-handdetector.tflite")

            FileLogger.i(TAG, "NPU: Creating NNAPI delegate...")
            delegate = NnApiDelegate()

            FileLogger.i(TAG, "NPU: Creating interpreter...")
            val options = Interpreter.Options()
                .addDelegate(delegate)
                .setNumThreads(4)

            interpreter = Interpreter(modelBuffer, options)
            FileLogger.i(TAG, "NPU: Interpreter created with NNAPI")

            return runBenchmark(interpreter, "NPU")

        } finally {
            interpreter?.close()
            delegate?.close()
        }
    }

    /**
     * Run benchmark with given interpreter
     */
    private fun runBenchmark(interpreter: Interpreter, backend: String): Float {
        FileLogger.i(TAG, "$backend: Creating tensors...")

        // Create input: [1, 256, 256, 3]
        val input = Array(1) {
            Array(256) {
                Array(256) {
                    FloatArray(3)
                }
            }
        }

        // Create outputs: boxes[1,2944,18] and scores[1,2944,1]
        val outputBoxes = Array(1) { Array(2944) { FloatArray(18) } }
        val outputScores = Array(1) { Array(2944) { FloatArray(1) } }
        val outputs = mapOf(0 to outputBoxes, 1 to outputScores)

        // Warmup
        FileLogger.i(TAG, "$backend: Running warmup (3 iterations)...")
        repeat(3) {
            interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        }

        // Benchmark
        FileLogger.i(TAG, "$backend: Running benchmark ($BENCHMARK_RUNS iterations)...")
        val times = mutableListOf<Float>()

        repeat(BENCHMARK_RUNS) {
            val start = System.nanoTime()
            interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
            val elapsed = (System.nanoTime() - start) / 1_000_000f
            times.add(elapsed)
        }

        val avg = times.average().toFloat()
        val min = times.minOrNull() ?: 0f
        val max = times.maxOrNull() ?: 0f

        FileLogger.i(TAG, "$backend: Complete - avg: ${avg}ms, min: ${min}ms, max: ${max}ms")

        return avg
    }

    /**
     * Generate smart recommendations
     */
    private fun generateRecommendations(
        deviceInfo: DeviceInfo,
        cpuWorking: Boolean,
        gpuWorking: Boolean,
        npuWorking: Boolean,
        cpuTime: Float,
        gpuTime: Float,
        npuTime: Float
    ): String {
        val report = StringBuilder()

        // Find fastest working option
        val options = mutableListOf<Pair<String, Float>>()
        if (cpuWorking) options.add("CPU" to cpuTime)
        if (gpuWorking) options.add("GPU" to gpuTime)
        if (npuWorking && npuTime < cpuTime * 0.9f) options.add("NPU" to npuTime)

        if (options.isEmpty()) {
            report.appendLine("✗ NO WORKING BACKENDS FOUND")
            report.appendLine()
            report.appendLine("This is critical - even CPU should work.")
            report.appendLine("Please check model files and TFLite version.")
            return report.toString()
        }

        val fastest = options.minByOrNull { it.second }!!

        report.appendLine("BEST OPTION: ${fastest.first}")
        report.appendLine("Performance: ${String.format("%.2f", fastest.second)}ms per inference")
        report.appendLine()

        when (fastest.first) {
            "GPU" -> {
                report.appendLine("✓ EXCELLENT! GPU acceleration is working!")
                report.appendLine()
                report.appendLine("Implementation:")
                report.appendLine("```kotlin")
                report.appendLine("val gpuDelegate = GpuDelegate()")
                report.appendLine("val options = Interpreter.Options().addDelegate(gpuDelegate)")
                report.appendLine("val interpreter = Interpreter(modelBuffer, options)")
                report.appendLine("```")
                report.appendLine()

                val fps = 1000f / (fastest.second * 2)  // 2 models in pipeline
                report.appendLine("Expected FPS for full pipeline: ${String.format("%.0f", fps)} FPS")

                if (deviceInfo.isQualcomm) {
                    report.appendLine()
                    report.appendLine("Note: Qualcomm Adreno GPU works excellently with TFLite!")
                }
            }

            "NPU" -> {
                report.appendLine("✓ GOOD! NPU (NNAPI) acceleration is working!")
                report.appendLine()
                report.appendLine("Implementation:")
                report.appendLine("```kotlin")
                report.appendLine("val nnApiDelegate = NnApiDelegate()")
                report.appendLine("val options = Interpreter.Options().addDelegate(nnApiDelegate)")
                report.appendLine("val interpreter = Interpreter(modelBuffer, options)")
                report.appendLine("```")
                report.appendLine()

                val fps = 1000f / (fastest.second * 2)
                report.appendLine("Expected FPS for full pipeline: ${String.format("%.0f", fps)} FPS")
            }

            "CPU" -> {
                report.appendLine("⚠ Using CPU (no hardware acceleration available)")
                report.appendLine()
                report.appendLine("Implementation:")
                report.appendLine("```kotlin")
                report.appendLine("val options = Interpreter.Options().setNumThreads(4)")
                report.appendLine("val interpreter = Interpreter(modelBuffer, options)")
                report.appendLine("```")
                report.appendLine()

                val fps = 1000f / (fastest.second * 2)
                report.appendLine("Expected FPS for full pipeline: ${String.format("%.0f", fps)} FPS")
                report.appendLine()

                if (deviceInfo.isExynos) {
                    report.appendLine("Note: For faster performance on Samsung Exynos:")
                    report.appendLine("  • Consider using Samsung Neural SDK for NPU access")
                    report.appendLine("  • Download: developer.samsung.com/neural")
                    report.appendLine("  • Expected: 4-10ms with Samsung Neural SDK")
                }

                report.appendLine()
                report.appendLine("To improve CPU performance:")
                report.appendLine("  • Quantize models to INT8 (2-4x faster)")
                report.appendLine("  • Optimize preprocessing pipeline")
                report.appendLine("  • Consider targeting Qualcomm devices (better acceleration)")
            }
        }

        report.appendLine()
        report.appendLine("─".repeat(60))
        report.appendLine("SUMMARY OF ALL OPTIONS:")
        report.appendLine()

        if (cpuWorking) {
            report.appendLine("CPU: ${String.format("%.2f", cpuTime)}ms ${if (fastest.first == "CPU") "← RECOMMENDED" else ""}")
        }
        if (gpuWorking) {
            report.appendLine("GPU: ${String.format("%.2f", gpuTime)}ms ${if (fastest.first == "GPU") "← RECOMMENDED" else ""}")
        }
        if (npuWorking) {
            val isActualNPU = npuTime < cpuTime * 0.9f
            report.appendLine("NPU: ${String.format("%.2f", npuTime)}ms ${if (!isActualNPU) "(CPU fallback)" else if (fastest.first == "NPU") "← RECOMMENDED" else ""}")
        }

        return report.toString()
    }

    /**
     * Load model file from assets
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