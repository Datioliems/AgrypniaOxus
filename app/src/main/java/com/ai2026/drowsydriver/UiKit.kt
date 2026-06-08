package com.ai2026.drowsydriver

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Bộ giao diện dùng chung (theme đỏ–đen "Stitch") cho màn Báo cáo & SOS.
 * Giúp các thẻ (card) bo góc, màu sắc và khoảng cách đồng nhất, đẹp hơn.
 */
object UiKit {
    const val BG = 0xFF0E0E10.toInt()
    const val CARD = 0xFF1B1B1E.toInt()
    const val CARD2 = 0xFF242428.toInt()
    const val RED = 0xFFFC1C46.toInt()
    const val GREEN = 0xFF2ECC71.toInt()
    const val AMBER = 0xFFF1C40F.toInt()
    const val TXT = 0xFFF5F5F7.toInt()
    const val SUB = 0xFF9A9AA0.toInt()

    fun dp(ctx: Context, v: Int): Int = (v * ctx.resources.displayMetrics.density).toInt()

    fun rounded(color: Int, radius: Float): GradientDrawable =
        GradientDrawable().apply { setColor(color); cornerRadius = radius }

    fun roundedStroke(fill: Int, stroke: Int, radius: Float, strokeW: Int): GradientDrawable =
        GradientDrawable().apply { setColor(fill); cornerRadius = radius; setStroke(strokeW, stroke) }

    /** Nền gradient (dùng cho hero card). */
    fun gradient(c1: Int, c2: Int, radius: Float): GradientDrawable =
        GradientDrawable(GradientDrawable.Orientation.TL_BR, intArrayOf(c1, c2)).apply { cornerRadius = radius }

    /** Thẻ bo góc theo chiều dọc. */
    fun card(ctx: Context, bg: Int = CARD): LinearLayout = LinearLayout(ctx).apply {
        orientation = LinearLayout.VERTICAL
        background = rounded(bg, dp(ctx, 20).toFloat())
        setPadding(dp(ctx, 18), dp(ctx, 16), dp(ctx, 18), dp(ctx, 16))
        layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(ctx, 12) }
    }

    fun title(ctx: Context, t: String, color: Int = TXT, size: Float = 22f): TextView =
        TextView(ctx).apply { text = t; setTextColor(color); textSize = size; setTypeface(typeface, android.graphics.Typeface.BOLD) }

    fun label(ctx: Context, t: String): TextView =
        TextView(ctx).apply { text = t; setTextColor(SUB); textSize = 13f; setPadding(0, dp(ctx, 2), 0, dp(ctx, 2)) }

    fun value(ctx: Context, t: String, color: Int = TXT, size: Float = 26f): TextView =
        TextView(ctx).apply { text = t; setTextColor(color); textSize = size; setTypeface(typeface, android.graphics.Typeface.BOLD) }

    /** Một dòng "nhãn → giá trị" dạng thẻ nhỏ (icon + tiêu đề + số liệu lớn). */
    fun statCard(ctx: Context, icon: String, titleText: String, valueText: String, accent: Int = TXT): LinearLayout {
        val c = card(ctx)
        val head = LinearLayout(ctx).apply { orientation = LinearLayout.HORIZONTAL }
        head.addView(TextView(ctx).apply { text = icon; textSize = 18f; setPadding(0, 0, dp(ctx, 8), 0) })
        head.addView(label(ctx, titleText).apply { textSize = 14f })
        c.addView(head)
        c.addView(value(ctx, valueText, accent).apply { setPadding(0, dp(ctx, 4), 0, 0) })
        return c
    }

    /** Thanh tiến trình bo góc (cho PERCLOS). */
    fun progressBar(ctx: Context, pct: Double, color: Int): View {
        val track = LinearLayout(ctx).apply {
            background = rounded(CARD2, dp(ctx, 8).toFloat())
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(ctx, 14))
                .apply { topMargin = dp(ctx, 8) }
        }
        val fillW = (pct.coerceIn(0.0, 100.0) / 100.0)
        val fill = View(ctx).apply {
            background = rounded(color, dp(ctx, 8).toFloat())
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, fillW.toFloat())
        }
        val spacer = View(ctx).apply {
            layoutParams = LinearLayout.LayoutParams(0, 1, (1.0 - fillW).toFloat())
        }
        track.orientation = LinearLayout.HORIZONTAL
        track.addView(fill); track.addView(spacer)
        return track
    }

    /** Chip nhỏ (badge) màu nền nhạt. */
    fun chip(ctx: Context, t: String, color: Int): TextView = TextView(ctx).apply {
        text = t; setTextColor(color); textSize = 13f
        gravity = Gravity.CENTER
        background = roundedStroke(Color.argb(38, Color.red(color), Color.green(color), Color.blue(color)), color, dp(ctx, 14).toFloat(), dp(ctx, 1))
        setPadding(dp(ctx, 14), dp(ctx, 6), dp(ctx, 14), dp(ctx, 6))
    }
}
