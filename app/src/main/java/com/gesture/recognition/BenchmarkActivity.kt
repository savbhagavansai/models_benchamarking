package com.gesture.recognition

import android.os.Bundle
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BenchmarkActivity : AppCompatActivity() {

    private lateinit var textView: TextView
    private lateinit var scrollView: ScrollView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            FileLogger.i("BenchmarkActivity", "=== BENCHMARK ACTIVITY STARTED ===")

            // Create simple UI programmatically
            scrollView = ScrollView(this).apply {
                layoutParams = android.view.ViewGroup.LayoutParams(
                    android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                    android.view.ViewGroup.LayoutParams.MATCH_PARENT
                )
                setPadding(32, 32, 32, 32)
                setBackgroundColor(android.graphics.Color.BLACK)
            }

            textView = TextView(this).apply {
                typeface = android.graphics.Typeface.MONOSPACE
                textSize = 10f
                setTextColor(android.graphics.Color.GREEN)
                text = "Running comprehensive benchmark...\n\nThis may take 1-2 minutes.\nPlease wait...\n\n"
            }

            scrollView.addView(textView)
            setContentView(scrollView)

            FileLogger.i("BenchmarkActivity", "UI created successfully")

            // Run benchmark in background
            runBenchmark()

        } catch (e: Exception) {
            FileLogger.e("BenchmarkActivity", "CRASH in onCreate: ${e.message}", e)
            e.printStackTrace()
            finish()
        }
    }

    private fun runBenchmark() {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                FileLogger.i("BenchmarkActivity", "Creating ModelBenchmark instance...")
                val benchmark = ModelBenchmark(applicationContext)

                FileLogger.i("BenchmarkActivity", "Starting benchmark execution...")
                textView.text = "Initializing benchmark...\n\nChecking models...\n\n"

                // Run on background thread
                val report = withContext(Dispatchers.Default) {
                    FileLogger.i("BenchmarkActivity", "Running benchmark on background thread...")
                    benchmark.runCompleteBenchmark()
                }

                FileLogger.i("BenchmarkActivity", "Benchmark completed!")

                // Display results
                textView.text = report

                // Save to file
                FileLogger.i("BENCHMARK", report)

            } catch (e: Exception) {
                val errorMsg = "Benchmark failed:\n\n${e.message}\n\n${e.stackTraceToString()}"
                FileLogger.e("BenchmarkActivity", "BENCHMARK CRASH: ${e.message}", e)
                textView.text = errorMsg
            }
        }
    }
}