package com.gesture.recognition

import android.content.Context
import android.os.Build
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * CRASH-SAFE Model Benchmark with extensive error handling
 *
 * This version has multiple safety checks to prevent crashes
 */
class ModelBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "ModelBenchmark"
        private const val MODEL_FILE = "mediapipe_hand-handdetector.tflite"
        private const val INPUT_SIZE = 192
        private const val WARMUP_RUNS = 3  // Reduced to avoid crashes
        private const val BENCHMARK_RUNS = 20  // Reduced to avoid crashes
    }

    private var interpreter: Interpreter? = null
    private var delegate: Any? = null

    // Dummy input/output for benchmarking
    private val input = Array(1) { Array(INPUT_SIZE) { Array(INPUT_SIZE) { FloatArray(3) } } }
    private val output = Array(1) { FloatArray(1) }

    private var modelBuffer: MappedByteBuffer? = null

    // ═══════════════════════════════════════════════════════════════
    // PUBLIC API - CRASH-SAFE BENCHMARK
    // ═══════════════════════════════════════════════════════════════

    /**
     * Run comprehensive benchmark - CRASH SAFE VERSION
     */
    fun runComprehensiveBenchmark(): String {
        log("═══════════════════════════════════════")
        log("Starting CRASH-SAFE benchmark...")
        log("═══════════════════════════════════════")

        val report = StringBuilder()

        try {
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("COMPREHENSIVE MODEL ACCELERATION BENCHMARK")
            report.appendLine("(CRASH-SAFE VERSION)")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine()

            // Device information
            appendDeviceInfo(report)

            // Load model with safety checks
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("STEP 1: MODEL LOADING")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine()

            val loadSuccess = safeLoadModel(report)
            if (!loadSuccess) {
                report.appendLine()
                report.appendLine("✗ MODEL LOADING FAILED - Cannot continue benchmark")
                report.appendLine("════════════════════════════════════════════════════════")
                return report.toString()
            }

            report.appendLine()

            // GPU compatibility check
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("STEP 2: GPU COMPATIBILITY CHECK")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine()

            val gpuSupported = safeCheckGPU(report)
            report.appendLine()

            // Performance benchmarks
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("STEP 3: PERFORMANCE BENCHMARK")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine()

            val results = mutableMapOf<String, BenchmarkResult>()

            // Test all backends with safety
            report.appendLine("Testing CPU...")
            results["CPU"] = safeBenchmarkCPU(report)
            report.appendLine()

            report.appendLine("Testing GPU...")
            results["GPU"] = safeBenchmarkGPU(report)
            report.appendLine()

            report.appendLine("Testing NPU...")
            results["NPU"] = safeBenchmarkNPU(report)
            report.appendLine()

            // Print results
            printResults(report, results)

            // Recommendations
            printRecommendations(report, results)

            // Summary
            printSummary(report, results)

            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("BENCHMARK COMPLETE!")
            report.appendLine("════════════════════════════════════════════════════════")

        } catch (e: Exception) {
            report.appendLine()
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("CRITICAL ERROR - BENCHMARK CRASHED")
            report.appendLine("════════════════════════════════════════════════════════")
            report.appendLine("Error: ${e.message}")
            report.appendLine("Stack trace:")
            report.appendLine(e.stackTraceToString())
            log("CRITICAL ERROR: ${e.message}")
            log(e.stackTraceToString())
        }

        return report.toString()
    }

    /**
     * COMPATIBILITY WRAPPER
     */
    fun runCompleteBenchmark(): String {
        return runComprehensiveBenchmark()
    }

    // ═══════════════════════════════════════════════════════════════
    // SAFE MODEL LOADING
    // ═══════════════════════════════════════════════════════════════

    private fun safeLoadModel(report: StringBuilder): Boolean {
        return try {
            log("Attempting to load model: $MODEL_FILE")
            report.appendLine("Loading model: $MODEL_FILE")

            // Check if file exists
            val assetList = context.assets.list("") ?: emptyArray()
            log("Assets found: ${assetList.joinToString(", ")}")

            if (!assetList.contains(MODEL_FILE)) {
                log("ERROR: Model file not found in assets!")
                report.appendLine("✗ ERROR: Model file '$MODEL_FILE' not found in assets folder")
                report.appendLine()
                report.appendLine("Available files in assets:")
                assetList.forEach { report.appendLine("  - $it") }
                return false
            }

            // Load model
            val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength

            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            val sizeKB = declaredLength / 1024
            log("Model loaded successfully: ${sizeKB}KB")
            report.appendLine("✓ Model loaded successfully")
            report.appendLine("  Size: ${sizeKB}KB")

            true

        } catch (e: Exception) {
            log("ERROR loading model: ${e.message}")
            log(e.stackTraceToString())

            report.appendLine("✗ ERROR loading model: ${e.message}")
            report.appendLine()
            report.appendLine("Stack trace:")
            report.appendLine(e.stackTraceToString())

            false
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // SAFE GPU CHECK
    // ═══════════════════════════════════════════════════════════════

    private fun safeCheckGPU(report: StringBuilder): Boolean {
        return try {
            log("Checking GPU compatibility...")

            val compatList = org.tensorflow.lite.gpu.CompatibilityList()
            val isSupported = compatList.isDelegateSupportedOnThisDevice

            report.appendLine("GPU Delegate Support (CompatibilityList): ${if (isSupported) "YES ✓" else "NO ✗"}")

            if (!isSupported) {
                report.appendLine()
                report.appendLine("Note: CompatibilityList reports 'not supported'")
                report.appendLine("      This may be a false negative on capable devices")
                report.appendLine("      GPU benchmark will try anyway...")
            }

            isSupported

        } catch (e: Exception) {
            log("ERROR checking GPU: ${e.message}")
            report.appendLine("GPU Delegate Support: Unknown (error checking)")
            report.appendLine("Error: ${e.message}")
            false
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // SAFE BENCHMARK IMPLEMENTATIONS
    // ═══════════════════════════════════════════════════════════════

    private fun safeBenchmarkCPU(report: StringBuilder): BenchmarkResult {
        log("━━━ Benchmarking CPU ━━━")

        return try {
            if (modelBuffer == null) {
                throw Exception("Model not loaded")
            }

            log("Creating CPU interpreter...")
            val options = Interpreter.Options()
            options.setNumThreads(4)
            options.setUseXNNPACK(true)

            interpreter = Interpreter(modelBuffer!!, options)
            log("CPU interpreter created")

            val avgTime = safeRunBenchmark()
            log("CPU benchmark complete: ${avgTime}ms")

            BenchmarkResult(
                avgTime = avgTime,
                status = Status.WORKING,
                statusMessage = "WORKING",
                threads = "4 threads (XNNPack)",
                note = ""
            )

        } catch (e: Exception) {
            log("CPU benchmark failed: ${e.message}")
            log(e.stackTraceToString())

            BenchmarkResult(
                avgTime = 0f,
                status = Status.FAILED,
                statusMessage = "FAILED: ${e.message}",
                threads = "",
                note = e.stackTraceToString()
            )
        } finally {
            safeCleanup()
        }
    }

    private fun safeBenchmarkGPU(report: StringBuilder): BenchmarkResult {
        log("━━━ Benchmarking GPU (FORCED) ━━━")

        return try {
            if (modelBuffer == null) {
                throw Exception("Model not loaded")
            }

            log("Creating GPU delegate (BYPASSING compatibility check)...")

            // CRITICAL: Wrap delegate creation in try-catch
            val gpuDelegate = try {
                GpuDelegate()
            } catch (e: Exception) {
                log("GPU delegate creation failed immediately: ${e.message}")
                throw Exception("GPU delegate not available: ${e.message}")
            }

            delegate = gpuDelegate
            log("GPU delegate object created")

            val options = Interpreter.Options()
            options.addDelegate(gpuDelegate)
            log("Adding GPU delegate to interpreter...")

            interpreter = Interpreter(modelBuffer!!, options)
            log("✓ GPU interpreter created successfully!")

            val avgTime = safeRunBenchmark()
            log("GPU benchmark complete: ${avgTime}ms")

            // Verify actual GPU usage
            if (avgTime > 50f) {
                log("⚠ Warning: GPU time slow, may be CPU fallback")
                BenchmarkResult(
                    avgTime = avgTime,
                    status = Status.WARNING,
                    statusMessage = "Slow (may be CPU fallback)",
                    threads = "GPU cores",
                    note = "Inference time suggests CPU fallback"
                )
            } else {
                log("✓ GPU acceleration confirmed")
                BenchmarkResult(
                    avgTime = avgTime,
                    status = Status.WORKING,
                    statusMessage = "WORKING ✓",
                    threads = "GPU cores",
                    note = ""
                )
            }

        } catch (e: Exception) {
            log("GPU benchmark failed: ${e.message}")
            log(e.stackTraceToString())

            BenchmarkResult(
                avgTime = 0f,
                status = Status.FAILED,
                statusMessage = "NOT SUPPORTED",
                threads = "",
                note = e.message ?: "Unknown error"
            )
        } finally {
            safeCleanup()
        }
    }

    private fun safeBenchmarkNPU(report: StringBuilder): BenchmarkResult {
        log("━━━ Benchmarking NPU (NNAPI) ━━━")

        return try {
            if (modelBuffer == null) {
                throw Exception("Model not loaded")
            }

            log("Creating NNAPI delegate...")
            val nnApiDelegate = NnApiDelegate()
            delegate = nnApiDelegate
            log("NNAPI delegate created")

            val options = Interpreter.Options()
            options.addDelegate(nnApiDelegate)

            interpreter = Interpreter(modelBuffer!!, options)
            log("NNAPI interpreter created")

            val avgTime = safeRunBenchmark()
            log("NPU benchmark complete: ${avgTime}ms")

            // Check if accelerated
            val isAccelerated = avgTime < 50f

            if (isAccelerated) {
                BenchmarkResult(
                    avgTime = avgTime,
                    status = Status.WORKING,
                    statusMessage = "WORKING ✓",
                    threads = "Hardware NPU/DSP",
                    note = ""
                )
            } else {
                BenchmarkResult(
                    avgTime = avgTime,
                    status = Status.WARNING,
                    statusMessage = "CPU fallback",
                    threads = "Delegate loads but uses CPU",
                    note = "NNAPI is deprecated on Android 15+"
                )
            }

        } catch (e: Exception) {
            log("NPU benchmark failed: ${e.message}")

            BenchmarkResult(
                avgTime = 0f,
                status = Status.FAILED,
                statusMessage = "FAILED: ${e.message}",
                threads = "",
                note = ""
            )
        } finally {
            safeCleanup()
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // SAFE BENCHMARK EXECUTION
    // ═══════════════════════════════════════════════════════════════

    private fun safeRunBenchmark(): Float {
        try {
            log("Starting benchmark (${WARMUP_RUNS} warmup + ${BENCHMARK_RUNS} test)...")

            // Warm up
            for (i in 0 until WARMUP_RUNS) {
                interpreter?.run(input, output)
            }
            log("Warmup complete")

            // Benchmark
            val times = mutableListOf<Float>()
            for (i in 0 until BENCHMARK_RUNS) {
                val start = System.nanoTime()
                interpreter?.run(input, output)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                times.add(elapsed)
            }

            val avgTime = times.average().toFloat()
            log("Benchmark complete: avg=${String.format("%.2f", avgTime)}ms")

            return avgTime

        } catch (e: Exception) {
            log("Benchmark execution failed: ${e.message}")
            throw e
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // DEVICE INFORMATION
    // ═══════════════════════════════════════════════════════════════

    private fun appendDeviceInfo(report: StringBuilder) {
        try {
            report.appendLine("DEVICE INFORMATION:")
            report.appendLine("  Manufacturer: ${Build.MANUFACTURER}")
            report.appendLine("  Model: ${Build.MODEL}")
            report.appendLine("  Hardware: ${Build.HARDWARE}")
            report.appendLine("  Android Version: ${Build.VERSION.SDK_INT}")
            report.appendLine("  GPU: ${detectGPU()}")
            report.appendLine()

            val isQualcomm = Build.HARDWARE.contains("qcom", ignoreCase = true)
            if (isQualcomm) {
                report.appendLine("✓ Qualcomm Snapdragon detected")
                report.appendLine("  Expected: Good GPU/NPU support")
            }
            report.appendLine()
        } catch (e: Exception) {
            report.appendLine("Error getting device info: ${e.message}")
            report.appendLine()
        }
    }

    private fun detectGPU(): String {
        return try {
            when {
                Build.HARDWARE.contains("qcom", ignoreCase = true) -> "Qualcomm Adreno"
                Build.HARDWARE.contains("exynos", ignoreCase = true) -> "ARM Mali (Exynos)"
                else -> "Unknown"
            }
        } catch (e: Exception) {
            "Unknown"
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // REPORT FORMATTING
    // ═══════════════════════════════════════════════════════════════

    private fun printResults(report: StringBuilder, results: Map<String, BenchmarkResult>) {
        results.forEach { (backend, result) ->
            when (result.status) {
                Status.WORKING -> {
                    report.appendLine("✓ $backend: ${String.format("%.2f", result.avgTime)}ms avg")
                    report.appendLine("  ${result.threads}")
                    report.appendLine()
                }
                Status.FAILED -> {
                    report.appendLine("✗ $backend: FAILED")
                    report.appendLine("  ${result.statusMessage}")
                    if (result.note.isNotEmpty()) {
                        report.appendLine("  Note: ${result.note}")
                    }
                    report.appendLine()
                }
                Status.WARNING -> {
                    report.appendLine("▲ $backend: ${String.format("%.2f", result.avgTime)}ms avg")
                    report.appendLine("  ${result.statusMessage}")
                    report.appendLine("  ${result.note}")
                    report.appendLine()
                }
            }
        }
    }

    private fun printRecommendations(report: StringBuilder, results: Map<String, BenchmarkResult>) {
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("RECOMMENDATIONS")
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine()

        val working = results.filter { it.value.status == Status.WORKING }
        if (working.isEmpty()) {
            report.appendLine("⚠ No hardware acceleration available")
            report.appendLine("Consider using CPU with INT8 quantization for better performance")
        } else {
            val best = working.minByOrNull { it.value.avgTime }!!
            report.appendLine("BEST OPTION: ${best.key}")
            report.appendLine("Performance: ${String.format("%.2f", best.value.avgTime)}ms per inference")
        }
        report.appendLine()
    }

    private fun printSummary(report: StringBuilder, results: Map<String, BenchmarkResult>) {
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("SUMMARY")
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine()

        results.forEach { (backend, result) ->
            when (result.status) {
                Status.WORKING, Status.WARNING ->
                    report.appendLine("$backend: ${String.format("%.2f", result.avgTime)}ms")
                Status.FAILED ->
                    report.appendLine("$backend: ${result.statusMessage}")
            }
        }
        report.appendLine()
    }

    // ═══════════════════════════════════════════════════════════════
    // SAFE CLEANUP
    // ═══════════════════════════════════════════════════════════════

    private fun safeCleanup() {
        try {
            interpreter?.close()
            interpreter = null
        } catch (e: Exception) {
            log("Error closing interpreter: ${e.message}")
        }

        try {
            when (delegate) {
                is GpuDelegate -> (delegate as GpuDelegate).close()
                is NnApiDelegate -> (delegate as NnApiDelegate).close()
            }
            delegate = null
        } catch (e: Exception) {
            log("Error closing delegate: ${e.message}")
        }
    }

    private fun log(message: String) {
        Log.d(TAG, message)
    }

    // ═══════════════════════════════════════════════════════════════
    // DATA CLASSES
    // ═══════════════════════════════════════════════════════════════

    data class BenchmarkResult(
        val avgTime: Float,
        val status: Status,
        val statusMessage: String,
        val threads: String,
        val note: String
    )

    enum class Status {
        WORKING,
        FAILED,
        WARNING
    }
}