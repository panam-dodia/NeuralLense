package com.panam.neurallens

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.*
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.min

enum class EnhancementMode {
    LOW_LIGHT,      // Zero-DCE (local)
    SHARPEN,        // OpenCV sharpening (local)
    DEBLUR,         // DACLIP via Replicate (cloud)
    BOTH            // Low-light + Sharpen (local)
}

class ImageEnhancer(context: Context) {
    private val zeroDCESession: OrtSession
    private val replicateClient: ReplicateClient
    private val env = OrtEnvironment.getEnvironment()

    // Progress callback for async operations
    var onProgress: ((Float, String) -> Unit)? = null

    companion object {
        private const val TAG = "ImageEnhancer"

        init {
            if (!OpenCVLoader.initDebug()) {
                Log.e(TAG, "OpenCV initialization failed")
            } else {
                Log.d(TAG, "OpenCV loaded successfully")
            }
        }
    }

    init {
        Log.d(TAG, "Loading Zero-DCE model...")
        val zeroDCEBytes = context.assets.open("zero_dce.onnx").readBytes()
        zeroDCESession = env.createSession(zeroDCEBytes)
        Log.d(TAG, "Zero-DCE loaded!")

        // Initialize Replicate client for cloud deblurring using BuildConfig
        replicateClient = ReplicateClient(BuildConfig.REPLICATE_API_TOKEN)
        Log.d(TAG, "Replicate client initialized")
    }

    /**
     * Enhance image based on selected mode.
     * Note: DEBLUR mode requires network and runs asynchronously.
     */
    fun enhance(bitmap: Bitmap, mode: EnhancementMode = EnhancementMode.BOTH): Bitmap {
        Log.d(TAG, "Enhancement mode: $mode")

        return when(mode) {
            EnhancementMode.LOW_LIGHT -> enhanceLowLight(bitmap)
            EnhancementMode.SHARPEN -> sharpenImage(bitmap)
            EnhancementMode.DEBLUR -> {
                // This should not be called directly - use deblurAsync instead
                Log.w(TAG, "DEBLUR mode called synchronously - returning original")
                bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
            }
            EnhancementMode.BOTH -> {
                val enhanced = enhanceLowLight(bitmap)
                val sharpened = sharpenImage(enhanced)
                enhanced.recycle()
                sharpened
            }
        }
    }

    /**
     * Deblur image using DACLIP via Replicate cloud API.
     * This is a suspend function - must be called from a coroutine.
     *
     * @param bitmap Input blurry image
     * @param maxSize Max dimension (smaller = faster, default 512)
     * @param onProgress Progress callback (0.0 to 1.0, status message)
     * @return Deblurred bitmap
     */
    suspend fun deblurAsync(
        bitmap: Bitmap,
        maxSize: Int = 512,
        onProgress: ((Float, String) -> Unit)? = null
    ): Bitmap {
        Log.d(TAG, "Starting cloud deblur via Replicate...")

        return try {
            replicateClient.restoreImage(
                bitmap = bitmap,
                maxSize = maxSize,
                onProgress = onProgress
            )
        } catch (e: ReplicateException) {
            Log.e(TAG, "Replicate error: ${e.message}")
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Deblur failed: ${e.message}")
            throw e
        }
    }

    fun enhanceLowLight(bitmap: Bitmap): Bitmap {
        val startTime = System.currentTimeMillis()

        val maxDim = 600
        val scale = min(maxDim.toFloat() / bitmap.width, maxDim.toFloat() / bitmap.height)
        val width = (bitmap.width * scale).toInt()
        val height = (bitmap.height * scale).toInt()

        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)

        val floatArray = FloatArray(1 * 3 * height * width)
        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            floatArray[i] = r
            floatArray[width * height + i] = g
            floatArray[2 * width * height + i] = b
        }

        Log.d(TAG, "Running Zero-DCE inference on ${width}x${height}...")

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floatArray),
            longArrayOf(1, 3, height.toLong(), width.toLong())
        )

        val results = zeroDCESession.run(mapOf("image" to inputTensor))
        val outputTensor = results[0].value as Array<Array<Array<FloatArray>>>

        val enhanced = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val outputPixels = IntArray(width * height)

        for (i in 0 until width * height) {
            val y = i / width
            val x = i % width

            val r = (outputTensor[0][0][y][x].coerceIn(0f, 1f) * 255).toInt()
            val g = (outputTensor[0][1][y][x].coerceIn(0f, 1f) * 255).toInt()
            val b = (outputTensor[0][2][y][x].coerceIn(0f, 1f) * 255).toInt()

            outputPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        enhanced.setPixels(outputPixels, 0, width, 0, 0, width, height)

        val result = Bitmap.createScaledBitmap(enhanced, bitmap.width, bitmap.height, true)

        enhanced.recycle()
        resized.recycle()
        inputTensor.close()

        val duration = System.currentTimeMillis() - startTime
        Log.d(TAG, "Zero-DCE completed in ${duration}ms")

        return result
    }

    fun sharpenImage(bitmap: Bitmap): Bitmap {
        val startTime = System.currentTimeMillis()

        try {
            val mat = Mat()
            Utils.bitmapToMat(bitmap, mat)

            val kernel = Mat(3, 3, CvType.CV_32F)
            kernel.put(0, 0,
                0.0, -1.0, 0.0,
                -1.0, 5.0, -1.0,
                0.0, -1.0, 0.0
            )

            val sharpened = Mat()
            Imgproc.filter2D(mat, sharpened, -1, kernel)

            val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(sharpened, result)

            mat.release()
            sharpened.release()
            kernel.release()

            val duration = System.currentTimeMillis() - startTime
            Log.d(TAG, "Sharpening completed in ${duration}ms")

            return result

        } catch (e: Exception) {
            Log.e(TAG, "Sharpening failed: ${e.message}")
            return bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
        }
    }

    fun release() {
        zeroDCESession.close()
    }
}