import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = CameraViewModel()

    var body: some View {
        ZStack(alignment: .topLeading) {
            CameraPreview(session: viewModel.session)
                .ignoresSafeArea()

            statusPanel
                .padding(.top, 24)
                .padding(.horizontal, 16)

            VStack {
                Spacer()
                Text(viewModel.footerMessage)
                    .font(.footnote)
                    .foregroundStyle(.white)
                    .padding(.vertical, 8)
                    .frame(maxWidth: .infinity)
                    .background(.black.opacity(0.45))
            }
            .ignoresSafeArea(edges: .bottom)
        }
        .task {
            await viewModel.start()
        }
    }

    private var statusPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle()
                    .fill(viewModel.status.level.color)
                    .frame(width: 12, height: 12)
                Text(viewModel.status.title)
                    .font(.headline)
                    .foregroundStyle(.white)
            }

            Text(viewModel.status.detail)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.9))

            HStack(spacing: 12) {
                metric("EAR", viewModel.status.ear)
                metric("MAR", viewModel.status.mar)
                metric("LAT", Float(viewModel.status.latencyMs))
            }

            Text("Source: \(viewModel.status.source)")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.75))
        }
        .padding(14)
        .frame(maxWidth: 330, alignment: .leading)
        .background(.black.opacity(0.72))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func metric(_ label: String, _ value: Float) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.65))
            Text(String(format: "%.2f", value))
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.white)
        }
    }
}

private extension AlertLevel {
    var color: Color {
        switch self {
        case .awake:
            return .green
        case .warning:
            return .yellow
        case .danger:
            return .red
        case .noFace:
            return .gray
        }
    }
}

#Preview {
    ContentView()
}

