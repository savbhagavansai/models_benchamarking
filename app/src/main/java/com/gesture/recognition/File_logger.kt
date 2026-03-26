package com.gesture.recognition

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * CRASH-SAFE File Logger
 *
 * Uses internal storage (no permissions needed)
 * Won't crash even if file operations fail
 *
 * Log location: /data/data/com.gesture.recognition/files/logs/debug_log.txt
 *
 * To retrieve log via ADB:
 * adb shell run-as com.gesture.recognition cat files/logs/debug_log.txt
 */
class FileLogger private constructor() {

    companion object {
        private const val TAG = "FileLogger"
        private var logFile: File? = null
        private var initialized = false

        /**
         * Initialize the file logger
         * CRASH-SAFE: Won't crash even if file operations fail
         */
        fun init(context: Context) {
            try {
                Log.d(TAG, "Initializing FileLogger...")

                // Use internal storage (no permissions needed)
                val logDir = File(context.filesDir, "logs")

                // Create directory if it doesn't exist
                if (!logDir.exists()) {
                    val created = logDir.mkdirs()
                    Log.d(TAG, "Log directory created: $created")
                }

                // Create log file
                logFile = File(logDir, "debug_log.txt")
                Log.d(TAG, "Log file path: ${logFile?.absolutePath}")

                // Write startup message
                val timestamp = getCurrentTime()
                logFile?.appendText("APP STARTED: $timestamp Log file: ${logFile?.absolutePath}\n")

                initialized = true
                Log.d(TAG, "✓ FileLogger initialized successfully")
                Log.d(TAG, "Log location: ${logFile?.absolutePath}")

            } catch (e: Exception) {
                Log.e(TAG, "FileLogger init failed (non-fatal): ${e.message}")
                Log.e(TAG, e.stackTraceToString())
                // Don't crash - just disable file logging
                initialized = false
                logFile = null
            }
        }

        /**
         * Log INFO level message
         * CRASH-SAFE: Won't crash even if file write fails
         */
        fun i(tag: String, message: String) {
            try {
                // Always log to Logcat
                Log.i(tag, message)

                // Try to log to file if initialized
                if (initialized && logFile != null) {
                    val timestamp = getCurrentTime()
                    logFile?.appendText("[$timestamp] [$tag] $message\n")
                }
            } catch (e: Exception) {
                Log.e(TAG, "File write failed (non-fatal): ${e.message}")
                // Don't crash - logging failure is not critical
            }
        }

        /**
         * Log DEBUG level message
         */
        fun d(tag: String, message: String) {
            try {
                Log.d(tag, message)
                if (initialized && logFile != null) {
                    val timestamp = getCurrentTime()
                    logFile?.appendText("[$timestamp] [DEBUG] [$tag] $message\n")
                }
            } catch (e: Exception) {
                Log.e(TAG, "File write failed: ${e.message}")
            }
        }

        /**
         * Log ERROR level message
         */
        fun e(tag: String, message: String) {
            try {
                Log.e(tag, message)
                if (initialized && logFile != null) {
                    val timestamp = getCurrentTime()
                    logFile?.appendText("[$timestamp] [ERROR] [$tag] $message\n")
                }
            } catch (e: Exception) {
                Log.e(TAG, "File write failed: ${e.message}")
            }
        }

        /**
         * Get current timestamp
         */
        private fun getCurrentTime(): String {
            return try {
                SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())
            } catch (e: Exception) {
                "timestamp_error"
            }
        }

        /**
         * Get log file path (for display purposes)
         */
        fun getLogPath(): String? {
            return logFile?.absolutePath
        }

        /**
         * Check if logger is initialized
         */
        fun isInitialized(): Boolean {
            return initialized
        }

        /**
         * Clear the log file
         */
        fun clear() {
            try {
                logFile?.writeText("")
                Log.d(TAG, "Log file cleared")
            } catch (e: Exception) {
                Log.e(TAG, "Clear failed: ${e.message}")
            }
        }
    }
}