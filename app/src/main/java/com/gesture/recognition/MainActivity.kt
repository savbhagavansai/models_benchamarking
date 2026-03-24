package com.gesture.recognition

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        FileLogger.init(this)
        FileLogger.i("MainActivity", "=== APP STARTED ===")

        // Switch back to BenchmarkActivity
        startActivity(Intent(this, BenchmarkActivity::class.java))

        finish()
    }
}