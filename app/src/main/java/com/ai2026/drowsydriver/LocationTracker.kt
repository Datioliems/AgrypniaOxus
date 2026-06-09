package com.ai2026.drowsydriver

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * Theo dõi GPS thời gian thực trong suốt phiên lái.
 *
 * Chiến lược:
 *  1. Thử GPS_PROVIDER trước (chính xác nhất)
 *  2. Fallback NETWORK_PROVIDER nếu GPS chưa lock hoặc trong nhà
 *  3. Mỗi 5 giây / 10 mét cập nhật 1 lần → đủ mịn cho lộ trình xe
 *
 * Sử dụng:
 *   LocationTracker.start(context)          // trong onCreate
 *   LocationTracker.stop()                  // trong onDestroy
 *   LocationTracker.latest                  // Location? — toàn cục
 *   LocationTracker.mapsUrl                 // "https://maps.google.com/?q=lat,lon"
 */
object LocationTracker {

    private const val TAG         = "LocationTracker"
    private const val MIN_TIME_MS = 5_000L   // 5 giây
    private const val MIN_DIST_M  = 10f      // 10 mét

    @Volatile var latest: Location? = null
        private set

    /** Google Maps link tới vị trí mới nhất — dùng cho SOS SMS + Tìm chỗ nghỉ */
    val mapsUrl: String
        get() = latest?.let { "https://maps.google.com/?q=${it.latitude},${it.longitude}" }
            ?: ""

    /** Có vị trí thực hay chưa */
    val hasLocation: Boolean get() = latest != null

    /** Tọa độ dạng geo URI cho Google Maps intent */
    val geoUri: String
        get() = latest?.let { "geo:${it.latitude},${it.longitude}" } ?: "geo:0,0"

    private var locationManager: LocationManager? = null
    private var isRunning = false

    private val listener = object : LocationListener {
        override fun onLocationChanged(loc: Location) {
            // Chỉ cập nhật nếu mới hơn hoặc chính xác hơn vị trí hiện tại
            if (isBetter(loc, latest)) {
                latest = loc
                Log.d(TAG, "GPS update: ${loc.latitude},${loc.longitude} " +
                        "acc=${loc.accuracy}m src=${loc.provider}")
            }
        }
        @Deprecated("Deprecated in Java")
        override fun onStatusChanged(p: String?, s: Int, e: Bundle?) {}
        override fun onProviderEnabled(p: String)  { Log.d(TAG, "Provider enabled: $p") }
        override fun onProviderDisabled(p: String) { Log.d(TAG, "Provider disabled: $p") }
    }

    fun start(context: Context) {
        if (isRunning) return
        val hasGps  = hasPerm(context, Manifest.permission.ACCESS_FINE_LOCATION)
        val hasNet  = hasPerm(context, Manifest.permission.ACCESS_COARSE_LOCATION)
        if (!hasGps && !hasNet) {
            Log.w(TAG, "No location permission — tracking disabled")
            return
        }
        val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        locationManager = lm
        isRunning = true

        // Gieo hạt nhanh bằng vị trí đã biết trước (passive)
        listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER,
               LocationManager.PASSIVE_PROVIDER).forEach { provider ->
            try {
                val last = lm.getLastKnownLocation(provider)
                if (last != null && isBetter(last, latest)) latest = last
            } catch (_: SecurityException) {}
        }

        // Đăng ký cập nhật thời gian thực
        if (hasGps && lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            try {
                lm.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER, MIN_TIME_MS, MIN_DIST_M,
                    listener, Looper.getMainLooper()
                )
                Log.i(TAG, "GPS_PROVIDER active")
            } catch (e: SecurityException) { Log.e(TAG, "GPS perm error: $e") }
        }
        if (hasNet && lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
            try {
                lm.requestLocationUpdates(
                    LocationManager.NETWORK_PROVIDER, MIN_TIME_MS, MIN_DIST_M,
                    listener, Looper.getMainLooper()
                )
                Log.i(TAG, "NETWORK_PROVIDER active")
            } catch (e: SecurityException) { Log.e(TAG, "Network perm error: $e") }
        }
    }

    fun stop() {
        try { locationManager?.removeUpdates(listener) } catch (_: Exception) {}
        locationManager = null
        isRunning = false
        Log.i(TAG, "Location tracking stopped")
    }

    /**
     * So sánh xem [candidate] có "tốt hơn" [current] không.
     * Tiêu chí: mới hơn 30 giây HOẶC chính xác hơn đáng kể.
     */
    private fun isBetter(candidate: Location, current: Location?): Boolean {
        if (current == null) return true
        val timeDelta = candidate.time - current.time
        if (timeDelta > 30_000) return true          // mới hơn 30s → luôn thắng
        if (timeDelta < -30_000) return false        // cũ hơn 30s → luôn thua
        return candidate.accuracy < current.accuracy // cùng thời điểm → chọn chính xác hơn
    }

    private fun hasPerm(ctx: Context, p: String) =
        ContextCompat.checkSelfPermission(ctx, p) == PackageManager.PERMISSION_GRANTED
}
