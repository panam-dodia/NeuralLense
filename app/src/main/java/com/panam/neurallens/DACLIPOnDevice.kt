package com.panam.neurallens

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random
import java.nio.LongBuffer

/**
 * On-Device DACLIP Image Restoration with Arm Optimization
 *
 * This class runs DACLIP inference locally on the device using:
 * - ONNX Runtime with NNAPI acceleration (Arm optimization)
 * - Reduced diffusion steps for faster processing
 * - Memory-efficient processing
 *
 * For Arm AI Hackathon 2025
 */
class DACLIPOnDevice(private val context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private var clipSession: OrtSession? = null
    private var unetSession: OrtSession? = null

    // CLIP preprocessing constants
    private val CLIP_MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
    private val CLIP_STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
    private val CLIP_SIZE = 224

    // SDE parameters (matching Python implementation)
    private val T = 100
    private val maxSigma = 50f / 255f
    private val eps = 0.005f
    private var thetas: FloatArray = FloatArray(0)
    private var thetasCumsum: FloatArray = FloatArray(0)
    private var dt: Float = 0f
    private var sigmas: FloatArray = FloatArray(0)
    private var sigmaBars: FloatArray = FloatArray(0)

    companion object {
        private const val TAG = "DACLIPOnDevice"
        private const val CLIP_MODEL = "models/daclip_encoder_int8.onnx"
        private const val UNET_MODEL = "models/unet_int8.onnx"
    }

    /**
     * Initialize models with Arm NNAPI optimization
     */
    fun initialize(): Boolean {
        return try {
            Log.d(TAG, "Initializing DACLIP models...")

            // Create session options with NNAPI (Arm acceleration)
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setInterOpNumThreads(4)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)

                // Enable NNAPI for Arm hardware acceleration
                try {
                    addNnapi()
                    Log.d(TAG, "✓ NNAPI (Arm acceleration) enabled")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI not available, using CPU: ${e.message}")
                }
            }

            // Load CLIP encoder
            Log.d(TAG, "Loading CLIP encoder...")
            val clipBytes = context.assets.open(CLIP_MODEL).readBytes()
            clipSession = env.createSession(clipBytes, sessionOptions)
            Log.d(TAG, "✓ CLIP loaded (${clipBytes.size / 1024 / 1024}MB)")

            // Force garbage collection before loading large UNet
            System.gc()
            Thread.sleep(500)

            // Load UNet
            Log.d(TAG, "Loading UNet...")
            val unetBytes = context.assets.open(UNET_MODEL).readBytes()
            unetSession = env.createSession(unetBytes, sessionOptions)
            Log.d(TAG, "✓ UNet loaded (${unetBytes.size / 1024 / 1024}MB)")

            // Initialize SDE schedule
            initSchedule()

            Log.d(TAG, "✓ DACLIP initialized successfully!")
            true

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OUT OF MEMORY! Models too large for this device.")
            Log.e(TAG, "Please use quantized models or cloud processing.")
            release()
            false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize: ${e.message}")
            e.printStackTrace()
            release()
            false
        }
    }

    /**
     * Initialize cosine noise schedule for SDE
     */
    private fun initSchedule() {
        val timesteps = T + 2
        val steps = timesteps + 1
        val s = 0.008f

        val x = FloatArray(steps) { it.toFloat() / timesteps }
        val alphasCumprod = FloatArray(steps) { i ->
            val val1 = (x[i] + s) / (1 + s) * Math.PI.toFloat() * 0.5f
            kotlin.math.cos(val1).let { it * it }
        }

        // Normalize
        val first = alphasCumprod[0]
        for (i in alphasCumprod.indices) {
            alphasCumprod[i] /= first
        }

        thetas = FloatArray(timesteps) { i -> 1 - alphasCumprod[i + 1] }

        // Cumsum
        thetasCumsum = FloatArray(timesteps)
        var sum = 0f
        for (i in thetas.indices) {
            sum += thetas[i]
            thetasCumsum[i] = sum - thetas[0]
        }

        dt = -1f / thetasCumsum.last() * ln(eps)

        sigmas = FloatArray(timesteps) { i ->
            sqrt(maxSigma * maxSigma * 2 * thetas[i])
        }

        sigmaBars = FloatArray(timesteps) { i ->
            sqrt(maxSigma * maxSigma * (1 - exp(-2 * thetasCumsum[i] * dt)))
        }

        Log.d(TAG, "SDE schedule initialized: T=$T, dt=$dt, max_sigma=$maxSigma")
    }

    /**
     * Restore degraded image
     *
     * @param bitmap Input degraded image
     * @param steps Number of diffusion steps (10-30 recommended for mobile)
     * @param maxSize Maximum dimension (256-512)
     * @param onProgress Progress callback
     * @return Restored image
     */
    fun restore(
        bitmap: Bitmap,
        steps: Int = 20,
        maxSize: Int = 384,
        onProgress: ((Int, Int, String) -> Unit)? = null
    ): Bitmap? {

        if (clipSession == null || unetSession == null) {
            Log.e(TAG, "Models not initialized!")
            return null
        }

        return try {
            val startTime = System.currentTimeMillis()

            // Resize if needed
            val processedBitmap = resizeIfNeeded(bitmap, maxSize)
            val width = processedBitmap.width
            val height = processedBitmap.height

            Log.d(TAG, "Starting restoration: ${width}x${height}, steps=$steps")
            onProgress?.invoke(0, steps, "Extracting features...")

            // Step 1: Extract CLIP features
            val (imageCtx, degraCtx) = extractFeatures(processedBitmap)
            onProgress?.invoke(1, steps, "Features extracted")

            // Step 2: Preprocess image
            val lq = bitmapToTensor(processedBitmap)

            // Step 3: Add initial noise
            val x = addNoise(lq)

            // Step 4: Reverse diffusion
            val sampleScale = T.toFloat() / steps

            for (step in steps downTo 1) {
                val t = (step * sampleScale).toInt().coerceIn(0, T - 1)

                onProgress?.invoke(steps - step + 1, steps,
                    "Restoring... (${steps - step + 1}/$steps)")

                // UNet prediction
                val noise = runUNet(x, lq, t, imageCtx, degraCtx, width, height)

                // SDE reverse step
                sdeReverseStep(x, noise, t, lq)

                // Periodic logging
                if (step % 5 == 0) {
                    val min = x.minOrNull() ?: 0f
                    val max = x.maxOrNull() ?: 0f
                    Log.d(TAG, "  Step ${steps - step + 1}/$steps: x range [$min, $max]")
                }
            }

            // Convert to bitmap
            val result = tensorToBitmap(x, width, height)

            // Resize back if needed
            val finalResult = if (width != bitmap.width || height != bitmap.height) {
                Bitmap.createScaledBitmap(result, bitmap.width, bitmap.height, true).also {
                    result.recycle()
                }
            } else {
                result
            }

            if (processedBitmap != bitmap) processedBitmap.recycle()

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "✓ Restoration complete in ${elapsed / 1000}s")
            onProgress?.invoke(steps, steps, "Complete!")

            finalResult

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OUT OF MEMORY during restoration")
            null
        } catch (e: Exception) {
            Log.e(TAG, "Restoration failed: ${e.message}")
            e.printStackTrace()
            null
        }
    }

    private fun resizeIfNeeded(bitmap: Bitmap, maxSize: Int): Bitmap {
        val maxDim = maxOf(bitmap.width, bitmap.height)
        return if (maxDim > maxSize) {
            val scale = maxSize.toFloat() / maxDim
            val newWidth = (bitmap.width * scale).toInt()
            val newHeight = (bitmap.height * scale).toInt()
            Log.d(TAG, "Resizing from ${bitmap.width}x${bitmap.height} to ${newWidth}x${newHeight}")
            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        } else {
            bitmap
        }
    }

    private fun extractFeatures(bitmap: Bitmap): Pair<FloatArray, FloatArray> {
        val resized = Bitmap.createScaledBitmap(bitmap, CLIP_SIZE, CLIP_SIZE, true)
        val pixels = IntArray(CLIP_SIZE * CLIP_SIZE)
        resized.getPixels(pixels, 0, CLIP_SIZE, 0, 0, CLIP_SIZE, CLIP_SIZE)

        val inputBuffer = FloatBuffer.allocate(3 * CLIP_SIZE * CLIP_SIZE)

        // Normalize with CLIP constants
        for (c in 0..2) {
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val value = when (c) {
                    0 -> ((pixel shr 16 and 0xFF) / 255f - CLIP_MEAN[0]) / CLIP_STD[0]
                    1 -> ((pixel shr 8 and 0xFF) / 255f - CLIP_MEAN[1]) / CLIP_STD[1]
                    else -> ((pixel and 0xFF) / 255f - CLIP_MEAN[2]) / CLIP_STD[2]
                }
                inputBuffer.put(value)
            }
        }
        inputBuffer.rewind()

        val inputTensor = OnnxTensor.createTensor(
            env, inputBuffer,
            longArrayOf(1, 3, CLIP_SIZE.toLong(), CLIP_SIZE.toLong())
        )

        val outputs = clipSession!!.run(mapOf("image" to inputTensor))
        val combined = (outputs[0].value as Array<FloatArray>)[0]

        inputTensor.close()
        outputs.forEach { it.value?.close() }
        resized.recycle()

        val imageCtx = combined.sliceArray(0 until 512)
        val degraCtx = combined.sliceArray(512 until 1024)

        return Pair(imageCtx, degraCtx)
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val tensor = FloatArray(3 * bitmap.height * bitmap.width)
        for (c in 0..2) {
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val idx = c * bitmap.width * bitmap.height + i
                tensor[idx] = when (c) {
                    0 -> (pixel shr 16 and 0xFF) / 255f
                    1 -> (pixel shr 8 and 0xFF) / 255f
                    else -> (pixel and 0xFF) / 255f
                }
            }
        }
        return tensor
    }

    private fun addNoise(tensor: FloatArray): FloatArray {
        val result = FloatArray(tensor.size)
        for (i in tensor.indices) {
            result[i] = tensor[i] + Random.nextFloat() * maxSigma * 2 - maxSigma
        }
        return result
    }

    private fun runUNet(
        x: FloatArray, lq: FloatArray, t: Int,
        imageCtx: FloatArray, degraCtx: FloatArray,
        width: Int, height: Int
    ): FloatArray {

        val noisyTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(x),
            longArrayOf(1, 3, height.toLong(), width.toLong())
        )
        val lqTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(lq),
            longArrayOf(1, 3, height.toLong(), width.toLong())
        )
        val timestepTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(longArrayOf(t.toLong())),
            longArrayOf(1)
        )
        val imageCtxTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(imageCtx),
            longArrayOf(1, 512)
        )
        val degraCtxTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(degraCtx),
            longArrayOf(1, 512)
        )

        val inputs = mapOf(
            "noisy_image" to noisyTensor,
            "lq_image" to lqTensor,
            "timestep" to timestepTensor,
            "image_context" to imageCtxTensor,
            "degra_context" to degraCtxTensor
        )

        val outputs = unetSession!!.run(inputs)
        val result = (outputs[0].value as Array<Array<Array<FloatArray>>>)[0]

        val noise = FloatArray(3 * height * width)
        for (c in 0..2) {
            for (y in 0 until height) {
                for (x in 0 until width) {
                    noise[c * height * width + y * width + x] = result[c][y][x]
                }
            }
        }

        inputs.values.forEach { it.close() }
        outputs.forEach { it.value?.close() }

        return noise
    }

    private fun sdeReverseStep(x: FloatArray, noise: FloatArray, t: Int, mu: FloatArray) {
        val score = FloatArray(noise.size) { -noise[it] / sigmaBars[t] }
        val theta = thetas[t]
        val sigma = sigmas[t]

        for (i in x.indices) {
            val reverseDrift = (theta * (mu[i] - x[i]) - sigma * sigma * score[i]) * dt
            val dispersion = sigma * Random.nextFloat() * sqrt(dt)
            x[i] = x[i] - reverseDrift - dispersion
        }
    }

    private fun tensorToBitmap(tensor: FloatArray, width: Int, height: Int): Bitmap {
        val pixels = IntArray(width * height)

        for (i in pixels.indices) {
            val r = (tensor[i].coerceIn(0f, 1f) * 255).toInt()
            val g = (tensor[width * height + i].coerceIn(0f, 1f) * 255).toInt()
            val b = (tensor[2 * width * height + i].coerceIn(0f, 1f) * 255).toInt()
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    fun release() {
        clipSession?.close()
        unetSession?.close()
        clipSession = null
        unetSession = null
        Log.d(TAG, "Models released")
    }
}