import AVFoundation
import Foundation

final class AlertSoundService {
    private var player: AVAudioPlayer?

    func playDanger() {
        AudioServicesPlaySystemSound(1005)
    }
}

