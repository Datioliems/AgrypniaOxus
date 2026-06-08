package com.ai2026.drowsydriver

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * Màn hình BÁO CÁO & PHÂN TÍCH — tổng quan mức độ buồn ngủ của phiên lái.
 * Thiết kế lại theo theme đỏ–đen "Stitch": hero card đánh giá rủi ro, các thẻ số liệu
 * bo góc, thanh PERCLOS, và trạng thái tự động gửi định vị.
 */
class AnalyticsActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val p = getSharedPreferences("agrypnia_prefs", MODE_PRIVATE)

        val sessionSec = p.getLong("session_sec", 0L)
        val drowsyEvents = p.getInt("drowsy_events", 0)
        val yawnEvents = p.getInt("yawn_events", 0)
        val drowsySec = p.getLong("drowsy_sec", 0L)
        val lastDrowsyAt = p.getString("last_drowsy_at", "—") ?: "—"
        val perclos = if (sessionSec > 0) drowsySec * 100.0 / sessionSec else 0.0
        val limit = EmergencyDispatcher.drowsyLimit(this)
        val autoSent = p.getBoolean("auto_sent_trip", false)
        val lastSentAt = p.getString("last_auto_sent_at", null)

        // Đánh giá mức rủi ro tổng thể
        val (riskTxt, riskColor, riskIcon) = when {
            perclos > 25 || drowsyEvents >= limit -> Triple("RỦI RO CAO", UiKit.RED, "🔴")
            perclos > 12 || drowsyEvents > 0 || yawnEvents >= 3 -> Triple("CẦN CHÚ Ý", UiKit.AMBER, "🟡")
            else -> Triple("AN TOÀN", UiKit.GREEN, "🟢")
        }

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(UiKit.dp(this@AnalyticsActivity, 16), UiKit.dp(this@AnalyticsActivity, 28),
                       UiKit.dp(this@AnalyticsActivity, 16), UiKit.dp(this@AnalyticsActivity, 28))
        }

        // Tiêu đề
        content.addView(UiKit.title(this, "📊  Báo cáo phiên lái", UiKit.TXT, 24f).apply {
            setPadding(0, 0, 0, UiKit.dp(this@AnalyticsActivity, 16))
        })

        // HERO — đánh giá rủi ro (gradient)
        val hero = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = UiKit.gradient(riskColor, 0xFF101012.toInt(), UiKit.dp(this@AnalyticsActivity, 24).toFloat())
            setPadding(UiKit.dp(this@AnalyticsActivity, 22), UiKit.dp(this@AnalyticsActivity, 22),
                       UiKit.dp(this@AnalyticsActivity, 22), UiKit.dp(this@AnalyticsActivity, 22))
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                .apply { bottomMargin = UiKit.dp(this@AnalyticsActivity, 16) }
        }
        hero.addView(TextView(this).apply { text = "MỨC ĐỘ AN TOÀN"; setTextColor(0xCCFFFFFF.toInt()); textSize = 13f })
        hero.addView(TextView(this).apply {
            text = "$riskIcon  $riskTxt"; setTextColor(0xFFFFFFFF.toInt()); textSize = 30f
            setTypeface(typeface, android.graphics.Typeface.BOLD); setPadding(0, UiKit.dp(this@AnalyticsActivity, 4), 0, 0)
        })
        hero.addView(TextView(this).apply {
            text = "Thời gian lái: ${fmt(sessionSec)}"; setTextColor(0xDDFFFFFF.toInt()); textSize = 14f
            setPadding(0, UiKit.dp(this@AnalyticsActivity, 8), 0, 0)
        })
        content.addView(hero)

        // Hàng 2 thẻ số liệu (buồn ngủ | ngáp)
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val s1 = UiKit.statCard(this, "😴", "Số lần buồn ngủ", "$drowsyEvents lần",
            if (drowsyEvents >= limit) UiKit.RED else if (drowsyEvents > 0) UiKit.AMBER else UiKit.GREEN)
        val s2 = UiKit.statCard(this, "🥱", "Số lần ngáp", "$yawnEvents lần", UiKit.TXT)
        (s1.layoutParams as LinearLayout.LayoutParams).apply { width = 0; weight = 1f; rightMargin = UiKit.dp(this@AnalyticsActivity, 6) }
        (s2.layoutParams as LinearLayout.LayoutParams).apply { width = 0; weight = 1f; leftMargin = UiKit.dp(this@AnalyticsActivity, 6) }
        row.addView(s1); row.addView(s2)
        content.addView(row)

        // PERCLOS card có thanh tiến trình
        val perclosCard = UiKit.card(this)
        perclosCard.addView(UiKit.label(this, "PERCLOS — tỉ lệ thời gian nhắm mắt").apply { textSize = 14f })
        perclosCard.addView(UiKit.value(this, String.format("%.1f%%", perclos),
            if (perclos > 15) UiKit.RED else UiKit.GREEN))
        perclosCard.addView(UiKit.progressBar(this, perclos, if (perclos > 15) UiKit.RED else UiKit.GREEN))
        perclosCard.addView(TextView(this).apply {
            text = "Ngưỡng cảnh báo khoa học: 15%"; setTextColor(UiKit.SUB); textSize = 12f
            setPadding(0, UiKit.dp(this@AnalyticsActivity, 6), 0, 0)
        })
        content.addView(perclosCard)

        // Thẻ thời gian buồn ngủ + lần gần nhất
        val timeCard = UiKit.card(this)
        timeCard.addView(UiKit.label(this, "Tổng thời gian buồn ngủ").apply { textSize = 14f })
        timeCard.addView(UiKit.value(this, fmt(drowsySec), if (drowsySec > 0) UiKit.AMBER else UiKit.TXT, 22f))
        timeCard.addView(UiKit.label(this, "Lần buồn ngủ gần nhất: $lastDrowsyAt").apply {
            setPadding(0, UiKit.dp(this@AnalyticsActivity, 8), 0, 0) })
        content.addView(timeCard)

        // Thẻ TỰ ĐỘNG GỬI ĐỊNH VỊ
        val autoCard = UiKit.card(this, if (autoSent) 0xFF2A1417.toInt() else UiKit.CARD)
        autoCard.addView(UiKit.title(this, "📍 Tự động gửi định vị", UiKit.TXT, 17f))
        autoCard.addView(TextView(this).apply {
            text = "Ngưỡng kích hoạt: $limit lần buồn ngủ / chuyến"; setTextColor(UiKit.SUB); textSize = 13f
            setPadding(0, UiKit.dp(this@AnalyticsActivity, 6), 0, UiKit.dp(this@AnalyticsActivity, 10))
        })
        autoCard.addView(UiKit.chip(this,
            if (autoSent) "✓ Đã gửi định vị lúc ${lastSentAt ?: "—"}" else "Chưa kích hoạt trong chuyến này",
            if (autoSent) UiKit.RED else UiKit.GREEN))
        content.addView(autoCard)

        // 🕐 Khung giờ hay buồn ngủ (histogram 24 giờ tích lũy)
        val hours = (p.getString("drowsy_hours", null)?.split(",")
            ?.map { it.trim().toIntOrNull() ?: 0 } ?: List(24) { 0 }).let {
            if (it.size >= 24) it.take(24) else it + List(24 - it.size) { 0 }
        }
        val totalH = hours.sum()
        val hourCard = UiKit.card(this)
        hourCard.addView(UiKit.title(this, "🕐 Khung giờ hay buồn ngủ", UiKit.TXT, 17f))
        if (totalH == 0) {
            hourCard.addView(TextView(this).apply {
                text = "Chưa đủ dữ liệu. Hệ thống sẽ tự học khung giờ bạn hay buồn ngủ qua các chuyến đi để khuyến cáo tránh lái xe vào giờ đó."
                setTextColor(UiKit.SUB); textSize = 13f
                setPadding(0, UiKit.dp(this@AnalyticsActivity, 8), 0, 0)
            })
        } else {
            val maxH = hours.maxOrNull() ?: 1
            val peak = hours.indexOf(maxH)
            hourCard.addView(TextView(this).apply {
                text = "Bạn hay buồn ngủ nhất vào khoảng %02d:00–%02d:00 (%d lần).".format(peak, (peak + 1) % 24, maxH)
                setTextColor(UiKit.TXT); textSize = 14f
                setPadding(0, UiKit.dp(this@AnalyticsActivity, 8), 0, UiKit.dp(this@AnalyticsActivity, 4))
            })
            hourCard.addView(TextView(this).apply {
                text = "⚠️ Nên tránh lái xe vào khung giờ này — hãy nghỉ ngơi đầy đủ hoặc đổi tài xế."
                setTextColor(UiKit.AMBER); textSize = 13f
                setPadding(0, 0, 0, UiKit.dp(this@AnalyticsActivity, 12))
            })
            val top = hours.withIndex().filter { it.value > 0 }.sortedByDescending { it.value }.take(5)
            for ((h, c) in top) {
                val frac = c.toFloat() / maxH
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
                    setPadding(0, UiKit.dp(this@AnalyticsActivity, 3), 0, UiKit.dp(this@AnalyticsActivity, 3))
                }
                row.addView(TextView(this).apply {
                    text = "%02d–%02dh".format(h, (h + 1) % 24)
                    setTextColor(if (h == peak) UiKit.RED else UiKit.SUB); textSize = 12f
                    layoutParams = LinearLayout.LayoutParams(UiKit.dp(this@AnalyticsActivity, 64), LinearLayout.LayoutParams.WRAP_CONTENT)
                })
                row.addView(View(this).apply {
                    background = UiKit.rounded(if (h == peak) UiKit.RED else UiKit.AMBER, UiKit.dp(this@AnalyticsActivity, 6).toFloat())
                    layoutParams = LinearLayout.LayoutParams(0, UiKit.dp(this@AnalyticsActivity, 12), frac)
                })
                row.addView(View(this).apply {
                    layoutParams = LinearLayout.LayoutParams(0, 1, 1f - frac)
                })
                row.addView(TextView(this).apply {
                    text = " $c"; setTextColor(UiKit.TXT); textSize = 12f
                    layoutParams = LinearLayout.LayoutParams(UiKit.dp(this@AnalyticsActivity, 34), LinearLayout.LayoutParams.WRAP_CONTENT)
                })
                hourCard.addView(row)
            }
        }
        content.addView(hourCard)

        // Khuyến cáo an toàn
        val tip = UiKit.card(this, UiKit.CARD2)
        tip.addView(TextView(this).apply {
            text = "💡 Khuyến cáo an toàn"; setTextColor(UiKit.AMBER); textSize = 15f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })
        tip.addView(TextView(this).apply {
            text = "PERCLOS > 15% hoặc nhiều lần buồn ngủ là dấu hiệu mệt mỏi nguy hiểm. " +
                   "Hãy dừng nghỉ 15–30 phút, uống nước hoặc đổi người lái trước khi tiếp tục hành trình."
            setTextColor(UiKit.SUB); textSize = 13f; setPadding(0, UiKit.dp(this@AnalyticsActivity, 6), 0, 0)
        })
        content.addView(tip)

        val scroll = ScrollView(this).apply {
            setBackgroundColor(UiKit.BG)
            addView(content)
        }
        setContentView(scroll)
    }

    private fun fmt(sec: Long): String {
        val h = sec / 3600; val m = (sec % 3600) / 60; val s = sec % 60
        return when {
            h > 0 -> "$h giờ $m phút"
            m > 0 -> "$m phút $s giây"
            else -> "$s giây"
        }
    }
}
