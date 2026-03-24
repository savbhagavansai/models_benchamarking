package com.gesture.recognition

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * Minimal MainActivity - Auto-launches benchmark on startup
 */
class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize logger
        FileLogger.init(this)

        // Auto-launch benchmark immediately
        startActivity(Intent(this, BenchmarkActivity::class.java))

        // Optional: finish MainActivity so back button doesn't return here
        finish()
    }
}