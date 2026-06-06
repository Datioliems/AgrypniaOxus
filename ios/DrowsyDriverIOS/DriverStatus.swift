import Foundation

enum AlertLevel {
    case awake
    case warning
    case danger
    case noFace
}

struct DriverStatus {
    var level: AlertLevel
    var title: String
    var detail: String
    var ear: Float
    var mar: Float
    var latencyMs: Int
    var source: String

    static let initial = DriverStatus(
        level: .noFace,
        title: "Initializing",
        detail: "Camera pipeline pending",
        ear: 0,
        mar: 0,
        latencyMs: 0,
        source: "iOS scaffold"
    )
}

