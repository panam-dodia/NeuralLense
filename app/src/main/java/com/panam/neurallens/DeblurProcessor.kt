package com.panam.neurallens

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class DeblurProcessor(private val context: Context) {
    private var daclipSession: OrtSession? = null
    private var unetSession: OrtSession? = null
    private val env = OrtEnvironment.getEnvironment()
    private var modelsLoaded = false

    companion object {
        private const val TAG = "DeblurProcessor"
        private const val DACLIP_MODEL = "models/daclip_int8.onnx"
        private const val UNET_MODEL = "models/unet_int8.onnx"

        private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        private const val CLIP_INPUT_SIZE = 224
        private const val MAX_PROCESS_SIZE = 256
    }

    private fun ensureModelsLoaded() {
        if (modelsLoaded) return

        try {
            Log.d(TAG, "Loading DACLIP models on-demand...")
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(4)

            Log.d(TAG, "Loading DA-CLIP...")
            val daclipBytes = context.assets.open(DACLIP_MODEL).readBytes()
            daclipSession = env.createSession(daclipBytes, sessionOptions)

            Log.d(TAG, "Loading UNet...")
            val unetBytes = context.assets.open(UNET_MODEL).readBytes()
            unetSession = env.createSession(unetBytes, sessionOptions)

            modelsLoaded = true
            Log.d(TAG, "DACLIP models loaded successfully!")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models: ${e.message}")
            e.printStackTrace()
            throw e
        }
    }

    fun deblur(bitmap: Bitmap, iterations: Int = 1): Bitmap? {
        val startTime = System.currentTimeMillis()

        return try {
            ensureModelsLoaded()

            val daclip = daclipSession ?: throw Exception("DACLIP not initialized")
            val unet = unetSession ?: throw Exception("UNet not initialized")

            val processedBitmap = resizeIfNeeded(bitmap)
            val width = processedBitmap.width
            val height = processedBitmap.height

            Log.d(TAG, "Processing ${width}x${height} image with $iterations iterations")

            val (imageContext, degraContext) = extractContexts(processedBitmap, daclip)
            val imageTensor = bitmapToTensor(processedBitmap)

            var current = imageTensor.copyOf()
            for (i in 0 until iterations) {
                val t = (iterations - i).toFloat() / iterations
                current = deblurStep(current, imageTensor, t, degraContext, imageContext, width, height, unet)
            }

            val result = tensorToBitmap(current, width, height)

            val finalResult = if (width != bitmap.width || height != bitmap.height) {
                Bitmap.createScaledBitmap(result, bitmap.width, bitmap.height, true).also {
                    result.recycle()
                }
            } else {
                result
            }

            if (processedBitmap != bitmap) processedBitmap.recycle()

            val duration = System.currentTimeMillis() - startTime
            Log.d(TAG, "Deblur completed in ${duration}ms")

            finalResult
        } catch (e: Exception) {
            Log.e(TAG, "Deblur failed: ${e.message}")
            e.printStackTrace()
            null
        }
    }

    private fun resizeIfNeeded(bitmap: Bitmap): Bitmap {
        val maxDim = maxOf(bitmap.width, bitmap.height)
        return if (maxDim > MAX_PROCESS_SIZE) {
            val scale = MAX_PROCESS_SIZE.toFloat() / maxDim
            val newWidth = (bitmap.width * scale).toInt()
            val newHeight = (bitmap.height * scale).toInt()
            Log.d(TAG, "Resizing from ${bitmap.width}x${bitmap.height} to ${newWidth}x${newHeight}")
            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        } else {
            bitmap
        }
    }

    private fun extractContexts(bitmap: Bitmap, session: OrtSession): Pair<FloatArray, FloatArray> {
        val resized = Bitmap.createScaledBitmap(bitmap, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, true)
        val pixels = IntArray(CLIP_INPUT_SIZE * CLIP_INPUT_SIZE)
        resized.getPixels(pixels, 0, CLIP_INPUT_SIZE, 0, 0, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE)

        val inputBuffer = FloatBuffer.allocate(3 * CLIP_INPUT_SIZE * CLIP_INPUT_SIZE)

        for (c in 0..2) {
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val value = when (c) {
                    0 -> ((pixel shr 16 and 0xFF) / 255f - MEAN[0]) / STD[0]
                    1 -> ((pixel shr 8 and 0xFF) / 255f - MEAN[1]) / STD[1]
                    else -> ((pixel and 0xFF) / 255f - MEAN[2]) / STD[2]
                }
                inputBuffer.put(value)
            }
        }
        inputBuffer.rewind()

        val inputTensor = OnnxTensor.createTensor(
            env, inputBuffer,
            longArrayOf(1, 3, CLIP_INPUT_SIZE.toLong(), CLIP_INPUT_SIZE.toLong())
        )

        val outputs = session.run(mapOf("input" to inputTensor))
        val imageContext = (outputs[0].value as Array<FloatArray>)[0]
        val degraContext = (outputs[1].value as Array<FloatArray>)[0]

        inputTensor.close()
        outputs.forEach { it.value?.close() }
        resized.recycle()

        return Pair(imageContext, degraContext)
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val tensor = FloatArray(3 * height * width)

        for (c in 0..2) {
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val idx = c * width * height + i
                tensor[idx] = when (c) {
                    0 -> (pixel shr 16 and 0xFF) / 255f
                    1 -> (pixel shr 8 and 0xFF) / 255f
                    else -> (pixel and 0xFF) / 255f
                }
            }
        }

        return tensor
    }

    private fun deblurStep(
        noisyImage: FloatArray,
        meanImage: FloatArray,
        timestep: Float,
        textContext: FloatArray,
        imageContext: FloatArray,
        width: Int,
        height: Int,
        session: OrtSession
    ): FloatArray {
        val noisyTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(noisyImage),
            longArrayOf(1, 3, height.toLong(), width.toLong())
        )
        val meanTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(meanImage),
            longArrayOf(1, 3, height.toLong(), width.toLong())
        )
        val timestepTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(floatArrayOf(timestep)),
            longArrayOf(1)
        )
        val textTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(textContext),
            longArrayOf(1, 512)
        )
        val imageTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(imageContext),
            longArrayOf(1, 512)
        )

        val inputs = mapOf(
            "noisy_image" to noisyTensor,
            "mean" to meanTensor,
            "timestep" to timestepTensor,
            "text_context" to textTensor,
            "image_context" to imageTensor
        )

        val outputs = session.run(inputs)
        val result = (outputs[0].value as Array<Array<Array<FloatArray>>>)[0]

        val flattened = FloatArray(3 * height * width)
        for (c in 0..2) {
            for (y in 0 until height) {
                for (x in 0 until width) {
                    flattened[c * height * width + y * width + x] = result[c][y][x]
                }
            }
        }

        inputs.values.forEach { it.close() }
        outputs.forEach { it.value?.close() }

        return flattened
    }

    private fun tensorToBitmap(tensor: FloatArray, width: Int, height: Int): Bitmap {
        val pixels = IntArray(width * height)

        for (i in pixels.indices) {
            val r = (tensor[i] * 255).toInt().coerceIn(0, 255)
            val g = (tensor[width * height + i] * 255).toInt().coerceIn(0, 255)
            val b = (tensor[2 * width * height + i] * 255).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    fun release() {
        daclipSession?.close()
        unetSession?.close()
    }
}