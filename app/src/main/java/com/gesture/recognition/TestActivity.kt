package com.gesture.recognition

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class TestActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val textView = TextView(this).apply {
            textSize = 12f
            setPadding(32, 32, 32, 32)
        }
        setContentView(textView)

        val report = StringBuilder()
        report.appendLine("=== MODEL FILE TEST ===\n")

        FileLogger.i("TestActivity", "Starting model file test...")

        // Test 1: Check if files exist
        report.appendLine("TEST 1: Checking if model files exist...\n")

        val models = listOf(
            "mediapipe_hand-handdetector.tflite",
            "mediapipe_hand-handlandmarkdetector.tflite"
        )

        for (modelName in models) {
            try {
                val fd = assets.openFd(modelName)
                val size = fd.length
                fd.close()

                report.appendLine("✓ $modelName")
                report.appendLine("  Size: ${size / 1024} KB\n")
                FileLogger.i("TestActivity", "✓ $modelName found (${size / 1024} KB)")

            } catch (e: Exception) {
                report.appendLine("✗ $modelName NOT FOUND!")
                report.appendLine("  Error: ${e.message}\n")
                FileLogger.e("TestActivity", "✗ $modelName not found: ${e.message}")
            }
        }

        // Test 2: Check memory
        report.appendLine("\nTEST 2: Memory Status\n")
        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory() / 1024 / 1024
        val freeMemory = runtime.freeMemory() / 1024 / 1024

        report.appendLine("Max Memory: ${maxMemory} MB")
        report.appendLine("Free Memory: ${freeMemory} MB\n")

        FileLogger.i("TestActivity", "Memory: ${freeMemory}MB free / ${maxMemory}MB max")

        // Test 3: Try loading one model
        report.appendLine("\nTEST 3: Try loading HandDetector model...\n")
        try {
            val modelBuffer = loadModelFile("mediapipe_hand-handdetector.tflite")
            report.appendLine("✓ Model loaded successfully!")
            report.appendLine("  Buffer size: ${modelBuffer.capacity() / 1024} KB\n")
            FileLogger.i("TestActivity", "✓ Model loaded successfully")

        } catch (e: Exception) {
            report.appendLine("✗ Failed to load model!")
            report.appendLine("  Error: ${e.message}\n")
            FileLogger.e("TestActivity", "✗ Failed to load: ${e.message}", e)
        }

        report.appendLine("\n=== TEST COMPLETE ===")
        textView.text = report.toString()

        FileLogger.i("TestActivity", "Test complete")
    }

    private fun loadModelFile(filename: String): java.nio.ByteBuffer {
        val fileDescriptor = assets.openFd(filename)
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