package com.panam.neurallens

import android.media.ExifInterface
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
private var daclipOnDevice: DACLIPOnDevice? = null

class MainActivity : AppCompatActivity() {

    private lateinit var replicateClient: ReplicateClient

    private lateinit var btnDAClip: Button
    // Views
    private lateinit var imageView: ImageView
    private lateinit var tvLabel: TextView
    private lateinit var tvProgress: TextView
    private lateinit var tvInfo: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var btnCamera: Button
    private lateinit var btnGallery: Button
    private lateinit var btnCompare: Button

    // State
    private var originalBitmap: Bitmap? = null
    private var enhancedBitmap: Bitmap? = null
    private var showingOriginal = true
    private lateinit var photoFile: File

    companion object {
        private const val TAG = "MainActivity"
        private const val MAX_IMAGE_DIMENSION = 1920
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            var bitmap = decodeSampledBitmapFromFile(photoFile.absolutePath)
            bitmap = getRotatedBitmap(bitmap, photoFile.absolutePath)
            originalBitmap?.recycle()  // Recycle OLD bitmap first
            originalBitmap = bitmap    // Then set the new rotated one
            imageView.setImageBitmap(originalBitmap)
            btnDAClip.isEnabled = true
            btnCompare.isEnabled = false
            enhancedBitmap?.recycle()
            enhancedBitmap = null
            tvLabel.text = "Original"
            showingOriginal = true
        }
    }

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            try {
                val inputStream = contentResolver.openInputStream(it)
                val loadedBitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (loadedBitmap != null) {
                    originalBitmap?.recycle()
                    originalBitmap = resizeBitmapIfNeeded(loadedBitmap)
                    imageView.setImageBitmap(originalBitmap)
                    btnDAClip.isEnabled = true
                    btnCompare.isEnabled = false
                    enhancedBitmap?.recycle()
                    enhancedBitmap = null
                    tvLabel.text = "Original"
                    showingOriginal = true
                } else {
                    Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                e.printStackTrace()
            }
        }
    }

    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            launchCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize Replicate client using BuildConfig
        replicateClient = ReplicateClient(BuildConfig.REPLICATE_API_TOKEN)

        // Find views
        imageView = findViewById(R.id.imageView)
        tvLabel = findViewById(R.id.tvLabel)
        tvProgress = findViewById(R.id.tvProgress)
        tvInfo = findViewById(R.id.tvInfo)
        progressBar = findViewById(R.id.progressBar)
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)
        btnCompare = findViewById(R.id.btnCompare)
        btnDAClip = findViewById(R.id.btnDAClip)

// Now you can use them
        btnDAClip.isEnabled = false
        btnCompare.isEnabled = false
        btnDAClip = findViewById(R.id.btnDAClip)

        photoFile = File(cacheDir, "photo.png")

        // Hide progress initially
        progressBar.visibility = View.GONE
        tvProgress.visibility = View.GONE

        // Button listeners
        btnCamera.setOnClickListener {
            checkCameraPermission()
        }

        btnGallery.setOnClickListener {
            pickImage.launch("image/*")
        }

        btnDAClip.setOnClickListener {
            enhanceWithDAClip()
        }

        btnCompare.setOnClickListener {
            toggleComparison()
        }
        lifecycleScope.launch(Dispatchers.IO) {
            daclipOnDevice = DACLIPOnDevice(this@MainActivity)
            val success = daclipOnDevice?.initialize()

            withContext(Dispatchers.Main) {
                if (success == true) {
                    Toast.makeText(this@MainActivity, "✓ On-device AI ready!", Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this@MainActivity, "⚠️ Models failed to load", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun decodeSampledBitmapFromFile(path: String): Bitmap {
        val options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeFile(path, options)

        val (width, height) = options.outWidth to options.outHeight
        var inSampleSize = 1

        if (width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
            val halfWidth = width / 2
            val halfHeight = height / 2

            while (halfWidth / inSampleSize >= MAX_IMAGE_DIMENSION &&
                halfHeight / inSampleSize >= MAX_IMAGE_DIMENSION) {
                inSampleSize *= 2
            }
        }

        return BitmapFactory.Options().apply {
            this.inSampleSize = inSampleSize
        }.let { BitmapFactory.decodeFile(path, it) }
    }
    private fun resizeBitmapIfNeeded(bitmap: Bitmap): Bitmap {
        val maxDim = MAX_IMAGE_DIMENSION
        return if (bitmap.width > maxDim || bitmap.height > maxDim) {
            val scale = maxDim.toFloat() / maxOf(bitmap.width, bitmap.height)
            val newWidth = (bitmap.width * scale).toInt()
            val newHeight = (bitmap.height * scale).toInt()
            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true).also {
                bitmap.recycle()
            }
        } else {
            bitmap
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED -> {
                launchCamera()
            }
            else -> {
                requestPermission.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun launchCamera() {
        val uri = FileProvider.getUriForFile(this, "$packageName.provider", photoFile)
        takePicture.launch(uri)
    }

    /**
     * Enhance image using DACLIP via Replicate cloud
     */
    private fun enhanceWithDAClip() {
        val bitmap = originalBitmap ?: return

        setProcessingState(true, "Connecting to AI server...")

        lifecycleScope.launch {
            try {
                val enhanced = withContext(Dispatchers.IO) {
                    replicateClient.restoreImage(
                        bitmap = bitmap,
                        maxSize = 1024,
                        onProgress = { progress, status ->
                            runOnUiThread {
                                progressBar.progress = (progress * 100).toInt()
                                tvProgress.text = status
                            }
                        }
                    )
                }

                enhancedBitmap?.recycle()
                enhancedBitmap = enhanced
                imageView.setImageBitmap(enhanced)
                tvLabel.text = "✨ Enhanced"
                showingOriginal = false

                setProcessingState(false)
                btnCompare.isEnabled = true

                Toast.makeText(this@MainActivity, "Enhancement complete!", Toast.LENGTH_SHORT).show()

            } catch (e: ReplicateException) {
                setProcessingState(false)
                Toast.makeText(
                    this@MainActivity,
                    "Error: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()

            } catch (e: Exception) {
                setProcessingState(false)
                Toast.makeText(
                    this@MainActivity,
                    "Network error. Please check your connection.",
                    Toast.LENGTH_LONG
                ).show()
                e.printStackTrace()
            }
        }
    }

    private fun setProcessingState(isProcessing: Boolean, message: String = "") {
        runOnUiThread {
            btnDAClip.isEnabled = !isProcessing
            btnCamera.isEnabled = !isProcessing
            btnGallery.isEnabled = !isProcessing

            if (isProcessing) {
                progressBar.visibility = View.VISIBLE
                tvProgress.visibility = View.VISIBLE
                progressBar.progress = 0
                tvProgress.text = message
            } else {
                progressBar.visibility = View.GONE
                tvProgress.visibility = View.GONE
            }
        }
    }

    private fun getRotatedBitmap(bitmap: Bitmap, path: String): Bitmap {
        val exif = ExifInterface(path)
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )

        val matrix = android.graphics.Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
        }

        return if (orientation != ExifInterface.ORIENTATION_NORMAL) {
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }
    private fun toggleComparison() {
        if (showingOriginal) {
            imageView.setImageBitmap(enhancedBitmap)
            tvLabel.text = "✨ Enhanced"
            btnCompare.text = "Show Original"
        } else {
            imageView.setImageBitmap(originalBitmap)
            tvLabel.text = "Original"
            btnCompare.text = "Show Enhanced"
        }
        showingOriginal = !showingOriginal
    }

    override fun onDestroy() {
        super.onDestroy()
        originalBitmap?.recycle()
        enhancedBitmap?.recycle()
    }
}