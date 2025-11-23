package com.panam.neurallens

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * Client for DACLIP image restoration via Replicate API.
 *
 * DACLIP automatically detects and fixes:
 * - Motion blur
 * - Haze/fog
 * - JPEG artifacts
 * - Low-light
 * - Noise
 * - Rain/raindrops
 * - Shadows
 * - Snow
 */
class ReplicateClient(private val apiToken: String) {

    companion object {
        private const val TAG = "ReplicateClient"
        private const val BASE_URL = "https://api.replicate.com/v1"

        // DACLIP model - correct version ID
        private const val DACLIP_VERSION = "5efbc85e1d0771704d74d81a660041abd270d48e99572badefbcdf81383f6af4"
        private val JSON_MEDIA_TYPE = "application/json".toMediaType()
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    /**
     * Restore a degraded image using DACLIP.
     * Automatically detects degradation type (blur, noise, low-light, etc.)
     */
    suspend fun restoreImage(
        bitmap: Bitmap,
        maxSize: Int = 512,
        onProgress: ((Float, String) -> Unit)? = null
    ): Bitmap = withContext(Dispatchers.IO) {

        Log.d(TAG, "Starting restoration: ${bitmap.width}x${bitmap.height}")
        onProgress?.invoke(0.05f, "Preparing image...")

        // Resize if needed
        val processedBitmap = resizeIfNeeded(bitmap, maxSize)

        // Convert to base64 data URI
        val base64Image = bitmapToDataUri(processedBitmap)
        Log.d(TAG, "Image encoded, size: ${base64Image.length} chars")

        onProgress?.invoke(0.1f, "Uploading to server...")

        // Step 1: Create prediction
        val predictionId = createPrediction(base64Image)
        Log.d(TAG, "Prediction created: $predictionId")

        onProgress?.invoke(0.15f, "AI is analyzing your image...")

        // Step 2: Poll for result
        val resultUrl = pollForResult(predictionId) { progress, status ->
            val mappedProgress = 0.15f + (progress * 0.75f)
            onProgress?.invoke(mappedProgress, status)
        }

        onProgress?.invoke(0.92f, "Downloading enhanced image...")

        // Step 3: Download result image
        val result = downloadImage(resultUrl)

        // Clean up
        if (processedBitmap != bitmap) {
            processedBitmap.recycle()
        }

        onProgress?.invoke(1.0f, "Complete!")
        Log.d(TAG, "Restoration complete: ${result.width}x${result.height}")

        result
    }
    private fun resizeIfNeeded(bitmap: Bitmap, maxSize: Int): Bitmap {
        val maxDim = maxOf(bitmap.width, bitmap.height)
        return if (maxDim > maxSize) {
            val scale = maxSize.toFloat() / maxDim
            val newWidth = (bitmap.width * scale).toInt()
            val newHeight = (bitmap.height * scale).toInt()
            Log.d(TAG, "Resizing from ${bitmap.width}x${bitmap.height} to ${newWidth}x${newHeight}")

            // Use higher quality scaling
            val matrix = android.graphics.Matrix()
            matrix.setScale(scale, scale)
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }

    private fun bitmapToDataUri(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
        val bytes = outputStream.toByteArray()
        val base64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
        return "data:image/png;base64,$base64"
    }

    private fun createPrediction(imageDataUri: String): String {
        val requestJson = JSONObject().apply {
            put("version", DACLIP_VERSION)
            put("input", JSONObject().apply {
                put("image", imageDataUri)
            })
        }

        val request = Request.Builder()
            .url("$BASE_URL/predictions")
            .post(requestJson.toString().toRequestBody(JSON_MEDIA_TYPE))
            .header("Authorization", "Token $apiToken")
            .header("Content-Type", "application/json")
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: "Unknown error"
            Log.e(TAG, "Create prediction failed: ${response.code} - $errorBody")
            throw ReplicateException("Failed to start enhancement: ${response.code}")
        }

        val responseJson = JSONObject(response.body!!.string())
        return responseJson.getString("id")
    }

    private suspend fun pollForResult(
        predictionId: String,
        onProgress: (Float, String) -> Unit
    ): String {
        var attempts = 0
        val maxAttempts = 120  // 10 minutes max

        while (attempts < maxAttempts) {
            val request = Request.Builder()
                .url("$BASE_URL/predictions/$predictionId")
                .get()
                .header("Authorization", "Token $apiToken")
                .build()

            val response = client.newCall(request).execute()

            if (!response.isSuccessful) {
                throw ReplicateException("Failed to check status: ${response.code}")
            }

            val responseJson = JSONObject(response.body!!.string())
            val status = responseJson.getString("status")

            Log.d(TAG, "Status: $status (attempt ${attempts + 1})")

            when (status) {
                "succeeded" -> {
                    val output = responseJson.get("output")
                    return when (output) {
                        is String -> output
                        else -> responseJson.getJSONArray("output").getString(0)
                    }
                }
                "failed" -> {
                    val error = responseJson.optString("error", "Unknown error")
                    throw ReplicateException("Enhancement failed: $error")
                }
                "canceled" -> {
                    throw ReplicateException("Enhancement was canceled")
                }
                "starting" -> {
                    onProgress(attempts.toFloat() / maxAttempts, "Starting AI model...")
                }
                "processing" -> {
                    onProgress(attempts.toFloat() / maxAttempts, "Enhancing image...")
                }
            }

            attempts++
            delay(5000)
        }

        throw ReplicateException("Timed out waiting for enhancement")
    }

    private fun downloadImage(url: String): Bitmap {
        val request = Request.Builder()
            .url(url)
            .get()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw ReplicateException("Failed to download result: ${response.code}")
        }

        val bytes = response.body!!.bytes()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: throw ReplicateException("Failed to decode result image")
    }
}

/**
 * Exception for Replicate API errors.
 */
class ReplicateException(message: String) : Exception(message)