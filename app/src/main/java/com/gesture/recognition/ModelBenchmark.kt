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
 * Model Benchmark - Tests CPU, GPU (FORCED), and NPU performance
 *
 * This version FORCES GPU delegate creation by BYPASSING the CompatibilityList check
 * which may incorrectly report "not supported" on capable Qualcomm Adreno devices.
 */
class ModelBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "ModelBenchmark"
        private const val MODEL_FILE = "mediapipe_hand-handdetector.tflite"
        private const val INPUT_SIZE = 192
        private const val WARMUP_RUNS = 5
        private const val BENCHMARK_RUNS = 50
    }

    private var interpreter: Interpreter? = null
    private var delegate: Any? = null

    // Dummy input/output for benchmarking
    private val input = Array(1) { Array(INPUT_SIZE) { Array(INPUT_SIZE) { FloatArray(3) } } }
    private val output = Array(1) { FloatArray(1) }

    private lateinit var modelBuffer: MappedByteBuffer

    // ═══════════════════════════════════════════════════════════════
    // PUBLIC API - COMPREHENSIVE BENCHMARK
    // ═══════════════════════════════════════════════════════════════

    /**
     * Run comprehensive benchmark testing CPU, GPU (forced), and NPU
     * Returns formatted report string
     */
    fun runComprehensiveBenchmark(): String {
        val report = StringBuilder()

        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("COMPREHENSIVE MODEL ACCELERATION BENCHMARK")
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine()

        // Device information
        appendDeviceInfo(report)

        // Load model
        modelBuffer = loadModelFile()

        // GPU compatibility check
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("STEP 1: GPU COMPATIBILITY CHECK")
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine()

        val gpuSupported = checkGPUCompatibilityList()
        report.appendLine("GPU Delegate Support: ${if (gpuSupported) "YES ✓" else "NO ✗"}")
        report.appendLine()

        if (!gpuSupported) {
            report.appendLine("The device reports GPU delegate is NOT supported.")
            report.appendLine("GPU benchmark will BYPASS this check and try anyway.")
        }
        report.appendLine()

        // Performance benchmarks
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("STEP 2: PERFORMANCE BENCHMARK (HandDetector Model)")
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine()

        val results = mutableMapOf<String, BenchmarkResult>()

        // Test all backends
        results["CPU"] = benchmarkCPU()
        results["GPU"] = benchmarkGPU()
        results["NPU"] = benchmarkNPU()

        // Print results
        printResults(report, results)

        // Recommendations
        printRecommendations(report, results)

        // Summary
        printSummary(report, results)

        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("BENCHMARK COMPLETE!")
        report.appendLine("════════════════════════════════════════════════════════")

        return report.toString()
    }

    /**
     * COMPATIBILITY WRAPPER
     * Maintains compatibility with existing BenchmarkActivity.kt
     * that calls runCompleteBenchmark()
     */
    fun runCompleteBenchmark(): String {
        return runComprehensiveBenchmark()
    }

    // ═══════════════════════════════════════════════════════════════
    // DEVICE INFORMATION
    // ═══════════════════════════════════════════════════════════════

    private fun appendDeviceInfo(report: StringBuilder) {
        report.appendLine("DEVICE INFORMATION:")
        report.appendLine("  Manufacturer: ${Build.MANUFACTURER}")
        report.appendLine("  Model: ${Build.MODEL}")
        report.appendLine("  Hardware: ${Build.HARDWARE}")
        report.appendLine("  Android Version: ${Build.VERSION.SDK_INT}")

        val gpuInfo = detectGPU()
        report.appendLine("  GPU: $gpuInfo")
        report.appendLine()

        // Device type detection
        val isQualcomm = Build.HARDWARE.contains("qcom", ignoreCase = true) ||
                         Build.HARDWARE.contains("qualcomm", ignoreCase = true)

        if (isQualcomm) {
            report.appendLine("✓ Qualcomm Snapdragon detected")
            report.appendLine("  Expected: Excellent GPU/NPU support")
        } else {
            report.appendLine("⚠ Non-Qualcomm device")
            report.appendLine("  Expected: Limited acceleration")
        }
        report.appendLine()
    }

    private fun detectGPU(): String {
        return try {
            when {
                Build.HARDWARE.contains("qcom", ignoreCase = true) ->
                    "Qualcomm Adreno"
                Build.HARDWARE.contains("exynos", ignoreCase = true) ->
                    "ARM Mali (Exynos)"
                Build.HARDWARE.contains("kirin", ignoreCase = true) ->
                    "ARM Mali (Kirin)"
                else ->
                    "Unknown"
            }
        } catch (e: Exception) {
            "Unknown"
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCHMARK IMPLEMENTATIONS
    // ═══════════════════════════════════════════════════════════════

    private fun benchmarkCPU(): BenchmarkResult {
        log("━━━ Benchmarking CPU ━━━")

        return try {
            val options = Interpreter.Options()
            options.setNumThreads(4)
            options.setUseXNNPACK(true)  // ARM NEON optimization

            interpreter = Interpreter(modelBuffer, options)

            log("CPU interpreter created")

            val avgTime = runBenchmark()

            log("✓ CPU: ${String.format("%.2f", avgTime)}ms avg")

            BenchmarkResult(
                avgTime = avgTime,
                status = Status.WORKING,
                statusMessage = "WORKING",
                threads = "4",
                note = ""
            )

        } catch (e: Exception) {
            log("✗ CPU failed: ${e.message}")
            BenchmarkResult(
                avgTime = 0f,
                status = Status.FAILED,
                statusMessage = e.message ?: "Unknown error",
                threads = "",
                note = ""
            )
        } finally {
            cleanup()
        }
    }

    private fun benchmarkGPU(): BenchmarkResult {
        log("━━━ Benchmarking GPU (FORCED - bypassing whitelist) ━━━")

        return try {
            // CRITICAL: FORCE GPU DELEGATE - BYPASS COMPATIBILITY CHECK
            log("Creating GPU delegate WITHOUT checking CompatibilityList...")
            log("This bypasses the false negative on Qualcomm devices")

            val gpuDelegate = GpuDelegate()
            delegate = gpuDelegate

            log("GPU delegate object created")

            val options = Interpreter.Options()
            options.addDelegate(gpuDelegate)

            log("Adding GPU delegate to interpreter options...")

            interpreter = Interpreter(modelBuffer, options)

            log("✓ GPU delegate created successfully!")
            log("✓ Interpreter initialized with GPU")

            val avgTime = runBenchmark()

            log("✓ GPU: ${String.format("%.2f", avgTime)}ms avg")

            // Verify it's actually using GPU (not CPU fallback)
            if (avgTime > 50f) {
                log("⚠ Warning: GPU time seems slow, may be CPU fallback")
            } else {
                log("✓ GPU acceleration confirmed (fast inference time)")
            }

            BenchmarkResult(
                avgTime = avgTime,
                status = Status.WORKING,
                statusMessage = "WORKING",
                threads = "GPU cores",
                note = ""
            )

        } catch (e: Exception) {
            log("✗ GPU failed: ${e.message}")
            log("Stack trace: ${e.stackTraceToString()}")

            BenchmarkResult(
                avgTime = 0f,
                status = Status.FAILED,
                statusMessage = "GPU not supported on this device",
                threads = "",
                note = e.message ?: "Unknown error"
            )
        } finally {
            cleanup()
        }
    }

    private fun benchmarkNPU(): BenchmarkResult {
        log("━━━ Benchmarking NPU (NNAPI) ━━━")

        return try {
            val nnApiDelegate = NnApiDelegate()
            delegate = nnApiDelegate

            log("NNAPI delegate created")

            val options = Interpreter.Options()
            options.addDelegate(nnApiDelegate)

            interpreter = Interpreter(modelBuffer, options)

            log("NNAPI interpreter created")

            val avgTime = runBenchmark()

            log("✓ NPU (NNAPI): ${String.format("%.2f", avgTime)}ms avg")

            // Check if actually accelerated
            val isAccelerated = avgTime < 50f

            if (isAccelerated) {
                log("✓ NPU acceleration detected (fast inference)")
                BenchmarkResult(
                    avgTime = avgTime,
                    status = Status.WORKING,
                    statusMessage = "WORKING ✓",
                    threads = "Hardware NPU/DSP",
                    note = ""
                )
            } else {
                log("⚠ NPU delegates loads but using CPU fallback")
                BenchmarkResult(
                    avgTime = avgTime,
                    status = Status.WARNING,
                    statusMessage = "CPU fallback (not accelerated)",
                    threads = "Delegate loads but uses CPU",
                    note = "NNAPI is deprecated on Android 15+"
                )
            }

        } catch (e: Exception) {
            log("✗ NPU failed: ${e.message}")
            BenchmarkResult(
                avgTime = 0f,
                status = Status.FAILED,
                statusMessage = e.message ?: "Unknown error",
                threads = "",
                note = ""
            )
        } finally {
            cleanup()
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // BENCHMARK EXECUTION
    // ═══════════════════════════════════════════════════════════════

    private fun runBenchmark(): Float {
        log("Starting benchmark (${WARMUP_RUNS} warmup + ${BENCHMARK_RUNS} test runs)...")

        // Warm up
        for (i in 0 until WARMUP_RUNS) {
            interpreter?.run(input, output)
        }

        log("Warmup complete, starting timed runs...")

        // Benchmark
        val times = mutableListOf<Float>()
        for (i in 0 until BENCHMARK_RUNS) {
            val start = System.nanoTime()
            interpreter?.run(input, output)
            val elapsed = (System.nanoTime() - start) / 1_000_000f
            times.add(elapsed)
        }

        val avgTime = times.average().toFloat()
        val minTime = times.minOrNull() ?: 0f
        val maxTime = times.maxOrNull() ?: 0f

        log("Benchmark complete: avg=${String.format("%.2f", avgTime)}ms, " +
            "min=${String.format("%.2f", minTime)}ms, " +
            "max=${String.format("%.2f", maxTime)}ms")

        return avgTime
    }

    // ═══════════════════════════════════════════════════════════════
    // REPORT FORMATTING
    // ═══════════════════════════════════════════════════════════════

    private fun printResults(
        report: StringBuilder,
        results: Map<String, BenchmarkResult>
    ) {
        results.forEach { (backend, result) ->
            when (result.status) {
                Status.WORKING -> {
                    report.appendLine("✓ $backend: ${String.format("%.2f", result.avgTime)}ms avg")
                    report.appendLine("  Threads: ${result.threads}")
                    report.appendLine("  Status: ${result.statusMessage}")
                    report.appendLine()
                }
                Status.FAILED -> {
                    report.appendLine("✗ $backend: FAILED")
                    report.appendLine("  Error: ${result.statusMessage}")
                    if (result.note.isNotEmpty()) {
                        report.appendLine("  Details: ${result.note}")
                    }
                    report.appendLine("  Status: NOT AVAILABLE")
                    report.appendLine()
                }
                Status.WARNING -> {
                    report.appendLine("▲ $backend: ${String.format("%.2f", result.avgTime)}ms avg")
                    report.appendLine("  Backend: ${result.statusMessage}")
                    report.appendLine("  Status: ${result.threads}")
                    report.appendLine("  Note: ${result.note}")
                    report.appendLine()
                }
            }
        }
    }

    private fun printRecommendations(
        report: StringBuilder,
        results: Map<String, BenchmarkResult>
    ) {
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("RECOMMENDATIONS")
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine()

        val bestBackend = results.filter { it.value.status == Status.WORKING }
            .minByOrNull { it.value.avgTime }

        if (bestBackend != null) {
            report.appendLine("BEST OPTION: ${bestBackend.key}")
            report.appendLine("Performance: ${String.format("%.2f", bestBackend.value.avgTime)}ms per inference")

            val fps = 1000 / (bestBackend.value.avgTime * 2)
            report.appendLine("Expected FPS for full pipeline: ${fps.toInt()} FPS")
            report.appendLine()

            report.appendLine("Implementation:")
            when (bestBackend.key) {
                "GPU" -> {
                    report.appendLine("```kotlin")
                    report.appendLine("val gpuDelegate = GpuDelegate()")
                    report.appendLine("val options = Interpreter.Options().addDelegate(gpuDelegate)")
                    report.appendLine("val interpreter = Interpreter(modelBuffer, options)")
                    report.appendLine("```")
                }
                "NPU" -> {
                    report.appendLine("```kotlin")
                    report.appendLine("val nnApiDelegate = NnApiDelegate()")
                    report.appendLine("val options = Interpreter.Options().addDelegate(nnApiDelegate)")
                    report.appendLine("val interpreter = Interpreter(modelBuffer, options)")
                    report.appendLine("```")
                }
                "CPU" -> {
                    report.appendLine("```kotlin")
                    report.appendLine("val options = Interpreter.Options()")
                    report.appendLine("    .setNumThreads(4)")
                    report.appendLine("    .setUseXNNPACK(true)")
                    report.appendLine("val interpreter = Interpreter(modelBuffer, options)")
                    report.appendLine("```")
                }
            }
        } else {
            report.appendLine("⚠ Using CPU (no hardware acceleration available)")
            report.appendLine()
            report.appendLine("To improve CPU performance:")
            report.appendLine("  • Quantize models to INT8 (2-4x faster)")
            report.appendLine("  • Optimize preprocessing pipeline")
            report.appendLine("  • Consider targeting Qualcomm devices (better acceleration)")
        }
        report.appendLine()
    }

    private fun printSummary(
        report: StringBuilder,
        results: Map<String, BenchmarkResult>
    ) {
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("SUMMARY OF ALL OPTIONS:")
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine()

        val cpuTime = results["CPU"]?.avgTime ?: 0f
        results.forEach { (backend, result) ->
            if (result.status == Status.WORKING) {
                val speedup = if (backend != "CPU" && cpuTime > 0) {
                    cpuTime / result.avgTime
                } else 1f

                val speedupText = if (speedup > 1) {
                    " ← ${String.format("%.0f", speedup)}x faster!"
                } else ""

                report.appendLine("$backend: ${String.format("%.2f", result.avgTime)}ms$speedupText")
            } else {
                report.appendLine("$backend: ${result.statusMessage}")
            }
        }
        report.appendLine()
    }

    // ═══════════════════════════════════════════════════════════════
    // HELPER FUNCTIONS
    // ═══════════════════════════════════════════════════════════════

    private fun checkGPUCompatibilityList(): Boolean {
        return try {
            val compatList = org.tensorflow.lite.gpu.CompatibilityList()
            compatList.isDelegateSupportedOnThisDevice
        } catch (e: Exception) {
            false
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun cleanup() {
        interpreter?.close()
        interpreter = null

        when (delegate) {
            is GpuDelegate -> (delegate as GpuDelegate).close()
            is NnApiDelegate -> (delegate as NnApiDelegate).close()
        }
        delegate = null
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