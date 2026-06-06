import CoreLocation
import Foundation

final class LocationAlertService: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()

    override init() {
        super.init()
        manager.delegate = self
    }

    func requestPermission() {
        manager.requestWhenInUseAuthorization()
    }

    func latestLocationText() -> String {
        guard let location = manager.location else {
            return "No location available"
        }
        return "\(location.coordinate.latitude), \(location.coordinate.longitude)"
    }
}

