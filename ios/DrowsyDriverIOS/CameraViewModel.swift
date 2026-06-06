import AVFoundation
import Foundation

@MainActor
final class CameraViewModel: NSObject, ObservableObject {
    @Published var status: DriverStatus = .initial
    @Published var footerMessage: String = "iOS scaffold - AVFoundation camera preview"

    let session = AVCaptureSession()

    private let inferenceService = DrowsinessInferenceService()
    private let alertSoundService = AlertSoundService()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "drowsy.ios.camera.frames")

    func start() async {
        let authorized = await requestCameraAccess()
        guard authorized else {
            status = DriverStatus(
                level: .noFace,
                title: "Camera permission needed",
                detail: "Enable camera permission in iOS Settings",
                ear: 0,
                mar: 0,
                latencyMs: 0,
                source: "iOS permission"
            )
            return
        }

        configureSession()
        session.startRunning()
        footerMessage = "iOS camera ready - inference placeholder"
    }

    private func requestCameraAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        default:
            return false
        }
    }

    private func configureSession() {
        guard session.inputs.isEmpty else { return }
        session.beginConfiguration()
        session.sessionPreset = .high

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        session.commitConfiguration()
    }
}

extension CameraViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let startedAt = Date()
        let result = inferenceService.analyze(sampleBuffer: sampleBuffer)
        let latencyMs = Int(Date().timeIntervalSince(startedAt) * 1000)

        Task { @MainActor in
            status = DriverStatus(
                level: result.level,
                title: result.title,
                detail: result.detail,
                ear: result.ear,
                mar: result.mar,
                latencyMs: latencyMs,
                source: result.source
            )

            if result.level == .danger {
                alertSoundService.playDanger()
            }
        }
    }
}

