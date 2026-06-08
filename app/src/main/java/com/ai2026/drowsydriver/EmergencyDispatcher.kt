package com.ai2026.drowsydriver

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Build
import android.telephony.SmsManager
import androidx.core.content.ContextCompat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Bộ gửi cảnh báo khẩn cấp — TỰ ĐỘNG lấy vị trí + gửi SMS cho người thân
 * khi tài xế vượt ngưỡng số lần buồn ngủ trong một chuyến đi.
 *
 * Khác với nút "gửi thử" (mở app SMS), hàm ở đây gửi trực tiếp qua SmsManager
 * (cần quyền SEND_SMS) nên không cần người dùng thao tác — phù hợp tình huống nguy hiểm.
 */
object EmergencyDispatcher {
    private const val PREFS = "agrypnia_prefs"
    const val DEFAULT_LIMIT = 3

    /** Ngưỡng số lần buồn ngủ tối đa trước khi tự gửi định vị (cấu hình ở màn SOS). */
    fun drowsyLimit(ctx: Context): Int =
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getInt("drowsy_limit", DEFAULT_LIMIT)

    private fun hasPerm(ctx: Context, p: String) =
        ContextCompat.checkSelfPermission(ctx, p) == PackageManager.PERMISSION_GRANTED

    private fun lastLocation(ctx: Context): String {
        if (!hasPerm(ctx, Manifest.permission.ACCESS_FINE_LOCATION) &&
            !hasPerm(ctx, Manifest.permission.ACCESS_COARSE_LOCATION)) return "(chưa cấp quyền vị trí)"
        return try {
            val lm = ctx.getSystemService(Context.LOCATION_SERVICE) as LocationManager
            val loc = lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                ?: lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            if (loc != null) "https://maps.google.com/?q=${loc.latitude},${loc.longitude}"
            else "(chưa lấy được GPS)"
        } catch (e: SecurityException) { "(lỗi quyền vị trí)" }
    }

    private fun smsManager(ctx: Context): SmsManager =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S)
            ctx.getSystemService(SmsManager::class.java)
        else @Suppress("DEPRECATION") SmsManager.getDefault()

    /**
     * Tự động gửi SMS định vị cho người thân đã lưu.
     * Trả về chuỗi kết quả để hiển thị (null nếu chưa cấu hình số điện thoại).
     */
    fun autoSendLocation(ctx: Context, reason: String): String? {
        val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val phone = (prefs.getString("contact_phone", "") ?: "").trim()
        if (phone.isBlank()) return null
        val name = prefs.getString("contact_name", "người thân") ?: "người thân"
        val time = SimpleDateFormat("HH:mm dd/MM", Locale.getDefault()).format(Date())
        val place = lastLocation(ctx)
        val msg = "AgrypniaOxus: Tài xế $reason lúc $time. Vị trí: $place. Hãy liên hệ kiểm tra ngay!"
        if (!hasPerm(ctx, Manifest.permission.SEND_SMS)) return "Chưa cấp quyền gửi SMS tự động"
        return try {
            val sm = smsManager(ctx)
            val parts = sm.divideMessage(msg)
            sm.sendMultipartTextMessage(phone, null, parts, null, null)
            prefs.edit().putString("last_auto_sent_at", time).apply()
            "Đã TỰ ĐỘNG gửi vị trí cho $name ($phone)"
        } catch (e: Exception) {
            "Lỗi gửi SMS tự động: ${e.message}"
        }
    }
}
