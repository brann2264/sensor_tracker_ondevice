//
//  ContentView.swift
//  SensorTracker
//
//  Created by Vasco Xu on 9/30/24.
//

import SwiftUI
import CoreMotion
import WatchConnectivity
import NearbyInteraction
import CoreBluetooth
import CoreML

import UnityFramework


struct ContentView: View {
    // device buttons
    @State private var isAirPodsSelected = false
    @State private var isPhoneSelected = false
    @State private var isWatchSelected = false
    
    @State private var showingUnity = false
        
    // helpers
    let nToMeasureFrequency = 100
    
    // labels
    @State private var socketStatusLabel = "N/A"
    @State private var phoneStatusLabel = "N/A"
    @State private var watchStatusLabel = "N/A"
    @State private var headphoneStatusLabel = "N/A"
    
    // networking fields
    @State private var socketIPField = "10.127.81.38"
    @State private var socketPortField = "8001"
    
    // unity variables
    @State private var animationTime: Float = 0.0


    // managers
    @State var phoneMotionManager: CMMotionManager!
    @State var phoneQueue = OperationQueue()
    @State var headphoneMotionManager: CMHeadphoneMotionManager!
    @State var headphoneQueue = OperationQueue()

    // watch manager
    @StateObject var sessionManager = SessionManager(socketClient: nil, sensorDataManager: nil)
    
    // phone
    @State var phoneCnt = 0
    @State var phonePrevTime: TimeInterval = NSDate().timeIntervalSince1970
    @State var phoneMeasuredFrequency: Double? = 25.0
    
    // watch
    @State var watchCnt = 0
    @State var watchPrevTime: TimeInterval = NSDate().timeIntervalSince1970
    @State var watchMeasuredFrequency: Double? = 25.0
        
    // headphone
    @State var headphoneCnt = 0
    @State var headphonePrevTime: TimeInterval = NSDate().timeIntervalSince1970
    @State var headphoneMeasuredFrequency: Double? = 25.0
    
    @State private var localSamplingRate: Double = 25.0 {
         didSet {
             // Update SessionManager's sampling rate
             sessionManager.watchSetFrequency = localSamplingRate
             // Send the updated sampling rate to the watch
             sessionManager.sendSamplingRate(localSamplingRate)
         }
     }
    
    @StateObject var sensorDataManager = SensorDataManager()
    @StateObject var mobilePoserManager = MobilePoserManager()
        
    init() {

    }
    
    // socket
    @State var socketClient: SocketClient?
    @State var client: StreamClient?
    
    @State private var directionsText: String = "Press Start Calibration to Begin"
    
    var body: some View {
        ZStack{
            Color(Color.black)
                .ignoresSafeArea()
            
            VStack() {
                Spacer()
                
                // networking stack
                VStack {
//                    Text("Network")
//                        .font(.largeTitle)
//                        .fontWeight(.medium)
//                        .foregroundStyle(.white)
//                    Divider()
//                        .background(.white)
//                        .padding([.leading, .trailing])
//                    
                    VStack {
                        HStack {
                            // socket text field
                            TextField("Socket...", text: $socketIPField)
                               .padding()
                               .background(Color.white)
                               .foregroundColor(.black)
                               .cornerRadius(5)
                               .padding([.horizontal], 15)
                            
                            // create socket button
                            Button(action: {
                                createSocketConnection()
                            }) {
                                Text("Create Socket")
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .padding(.vertical, 15)
                            }
                            .padding([.horizontal], 5)
                            .frame(width: 140)
                            .background(Color.blue)
                            .cornerRadius(10)
                        }
                        
                        HStack {
                            // port field
                            TextField("Port...", text: $socketPortField)
                               .padding()
                               .background(Color.white)
                               .foregroundColor(.black)
                               .cornerRadius(5)
                               .padding([.horizontal], 15)
                            
                            // stop socket button
                            Button(action: {
                                stopSocketConnection()
                            }) {
                                Text("Stop Socket")
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .padding(.vertical, 15)
                            }
                            .padding([.horizontal], 15)
                            .frame(width: 140)
                            .background(Color.red)
                            .cornerRadius(10)
                        }
                        
                        // socket status label
                        HStack {
                            Label("Socket Status: ", systemImage: "")
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                            Spacer()
                            Label(self.socketStatusLabel, systemImage: "")
                                .foregroundColor(.white)
                        }.padding()
                        
                        Button(action: {
                            Task {
                                    await startCalibration()
                            }
//                            startCalibration()
                        }) {
                            Text("Start Calibration")
                                .foregroundColor(.white)
                        }
                        .buttonBorderShape(.roundedRectangle(radius: 10))
                        
                        
                        // Unity visualizer button
                        Button(action: {
                            launchUnity()
                        }) {
                            Text("Unity Visualizer")
                                .foregroundColor(.white)
                        }
                        .buttonBorderShape(.roundedRectangle(radius: 10))
                                                
                    }.padding()
                }
                
                Spacer()
                
                Text("\(self.directionsText)")
                        .foregroundColor(.secondary)
                
                Spacer()
                
                VStack(alignment: .leading) {
                    Text("Sampling Rate: \(Int(localSamplingRate)) Hz")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Slider(value: $localSamplingRate, in: 10...100, step: 5)
                        .accentColor(.blue)
                        .padding(.horizontal)
                    
                    // Optionally, display the current sampling rate
                    Text("Current Sampling Rate: \(Int(localSamplingRate)) Hz")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }.padding()
                
                Spacer()
                
                Text("Devices")
                    .font(.largeTitle)
                    .fontWeight(.medium)
                    .foregroundStyle(.white)
                Divider()
                    .background(.white)
                    .padding([.leading, .trailing])
                
                VStack {
                    HStack(spacing: 40.0) {
                        Spacer()
                        // airpods button
                        Button(action: {
                            self.isAirPodsSelected.toggle()
                            toggleHeadphoneMotion()
                        }) {
                            Image(systemName: "airpods.gen3")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .foregroundColor(isAirPodsSelected ? .blue : .white)
                        }
                        // phone button
                        Button(action: {
                            self.isPhoneSelected.toggle()
                            togglePhone()
                            
//                            if client.isConnected{
//                                client.stop()
//                            }
//                            else{
//                                client.start()
//                            }
                            
                        }) {
                            Image(systemName: "iphone.gen3")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .foregroundColor(isPhoneSelected ? .blue : .white)
                        }
                        // watch button
                        Button(action: {
                            self.isWatchSelected.toggle()
                            toggleWatch()
                        }) {
                            Image(systemName: "applewatch")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .foregroundColor(isWatchSelected ? .blue : .white)
                        }
                        Spacer()
                    }
                    VStack {
                        HStack {
                            Label("Phone Status:", systemImage: "")
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                            Spacer()
                            Label(self.phoneStatusLabel, systemImage: "")
                                .foregroundColor(.white)
                        }
                        HStack {
                            Label("Watch Status: ", systemImage: "")
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                            Spacer()
                            Label(sessionManager.watchStatusLabel, systemImage: "")
                                .foregroundColor(.white)
                        }
                        HStack {
                            Label("Headphone Status: ", systemImage: "")
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                            Spacer()
                            Label(self.headphoneStatusLabel, systemImage: "")
                                .foregroundColor(.white)
                        }
                    }.padding()

                }.padding()

                Spacer()
            }
            .onAppear {
                // Load sampling rate from UserDefaults
                let savedRate = UserDefaults.standard.double(forKey: "samplingRate")
                self.localSamplingRate = savedRate > 0 ? savedRate : 25.0
            }
            .onTapGesture {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
            .ignoresSafeArea()
        }
    }

    // Update the sendDataToUnity function to use a faster-changing time
    private func sendDataToUnity(currPose: MLMultiArray, rootPos: MLMultiArray) {
        var poseData = [Float](repeating: 0, count: currPose.count)
        var translationData = [Float](repeating: 0, count: rootPos.count)
        
        for i in 0..<currPose.count {
            poseData[i] = currPose[i].floatValue
        }
        
        for i in 0..<rootPos.count {
            translationData[i] = rootPos[i].floatValue
        }
        
        // Format data as expected by Unity: "pose1,pose2,...,pose72#trans1,trans2,trans3$"
        let poseString = poseData.map { String(format: "%.6f", $0) }.joined(separator: ",")
        let translationString = translationData.map { String(format: "%.6f", $0) }.joined(separator: ",")
        let messageString = "\(poseString)#\(translationString)$"
                
        // Send directly to Unity GameObject
        UnityFramework.getInstance()?.sendMessageToGO(
            withName: "Scripts",
            functionName: "ReceivePoseData",
            message: messageString
        )
    }

    private func launchUnity() {
        UnityManager.getInstance().show()
        showingUnity = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.startUnityDataStream()
        }
    }
    
    private func startCalibration() async {
        
        print("Put imu 1 aligned with your body reference frame (x = Left, y = Up, z = Forward)")
        self.directionsText = "Put imu 1 aligned with your body reference frame (x = Left, y = Up, z = Forward)"
        
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        
        print("Keep for 3 seconds ...")
        self.directionsText = "Keep for 3 seconds ..."
        let measurements = sensorDataManager.getMeanMeasurement(numSeconds: 3, bufferLen: 40)
        let oris = measurements.meanQuaternions[0]!
        
        print("Wear all imus correctly")
        self.directionsText = "\tFinish.\nWear all imus correctly"
        
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        
        print("\rStand straight in T-pose and be ready. The calibration will begin after 3 seconds.")
        
        try? await Task.sleep(nanoseconds: 3_000_000_000)
        
        self.directionsText = "\rStand straight in T-pose and be ready. The calibration will begin after 3 seconds."
        
        let measurements2 = sensorDataManager.getMeanMeasurement(numSeconds: 3, bufferLen: 40)
        mobilePoserManager.calibrate(imu1_ori: oris, accMeans: measurements2.meanAccelerations, oriMeans: measurements2.meanQuaternions)
        
        print("Calibration done. Access Unity Visualize to see the pose estimations")
        self.directionsText = "Calibration done. Access Unity Visualize to see the pose estimations"
        launchUnity()
    }



    private func startUnityDataStream() {
        Timer.scheduledTimer(withTimeInterval: 1.0/30.0, repeats: true) { timer in
            guard showingUnity else {
                timer.invalidate()
                return
            }
            
            let data = sensorDataManager.getMostRecentIMU()
 
//            let model_outputs = client?.mlManager.predict(ori_raw: data.Ori, acc_raw: data.Acc)
            let model_outputs = mobilePoserManager.predict(ori_raw: data.Ori, acc_raw: data.Acc)
            
            if model_outputs != nil {
                sendDataToUnity(currPose: model_outputs!.pose, rootPos: model_outputs!.rootPos)
            }
            
//            let preds = client!.mlManager.getPredictions()
            
//            if preds.pose != nil && preds.rootPos != nil {
//                print("Sending data to Unity")
//                sendDataToUnity(currPose: preds.pose!, rootPos: preds.rootPos!)
//            }
            
        }
    }
    
    func initData() -> MLMultiArray {
        let shape: [NSNumber] = [1, 40, 24]
        guard let data = try? MLMultiArray(shape: shape, dataType: .double) else {
            fatalError("Error creating MLMultiArray.")
        }
        let length = data.count
        for i in 0..<length {
            data[i] = 0
        }
        return data
    }
    
    func togglePhone() {
        print("Toggle Phone...")
        if self.isPhoneSelected {
            startPhoneMotion()
        } else {
            stopPhoneMotion()
        }
    }
    
    func toggleWatch() {
        print("Toggle Watch...")
        if self.isWatchSelected {
            pairWatch()
        } else {
            stopWatchMotion()
        }
    }
    
    func toggleHeadphoneMotion() {
        print("Toggle Headphone...")
        if self.isAirPodsSelected {
            startHeadphoneMotion()
        } else {
            stopHeadphoneMotion()
        }
    }
    
    
    private func updateSocketStatusLabel(status: Bool) {
        DispatchQueue.main.async {
            if status {
                self.socketStatusLabel = "Ready!"
            } else {
                self.socketStatusLabel = "Not ready..."
            }
        }
    }
    
    func createSocketConnection() {
        let ip = socketIPField as String? ?? "0.0.0.0"
        let port = UInt16(socketPortField as String? ?? "0") ?? 8000
        let deviceIdentifier = "left"
        
        print("try to connect \(ip):\(port) with device id \(deviceIdentifier)")
        socketClient = SocketClient(ip: ip, portInt: port, deviceID: deviceIdentifier) { (status) in
            self.updateSocketStatusLabel(status: status)
        }
        
        client = StreamClient(host: ip, port: 8890)
        
        client?.start()
        
        // update socket client of sessionManager
        self.sessionManager.socketClient = socketClient
    }
    
    func restartSocketConnection() {
        guard let socketClient = socketClient else {
            return
        }
        socketClient.restart() { (status) in
            self.updateSocketStatusLabel(status: status)
        }
        socketClient.send(text: "restarted")
    }
    
    func stopSocketConnection() {
        guard let socketClient = socketClient else {
            return
        }
        socketClient.stop()
    }
    
    func pairWatch() {
        print("watch status")
        print("is watch app installed: ", WCSession.default.isWatchAppInstalled)
        print("is paired: ", WCSession.default.isPaired)
        print("is reachable: ", WCSession.default.isReachable)
        
        self.sessionManager.sensorDataManager = self.sensorDataManager
        
        if (WCSession.default.isReachable) {
            do {
                try WCSession.default.updateApplicationContext(["samplingRate": localSamplingRate])
                watchStatusLabel = "Connected!"
                
                // send IP address to watch
                sendIPToWatch(ip: socketIPField)
                
                // start watch motion when connected
                startWatchMotion()
            }
            catch {
                print(error)
            }
        } else {
            watchStatusLabel = "Not reachable..."
        }
    }
    
    func sendIPToWatch(ip: String) {
        if (WCSession.default.isReachable) {
            do {
                try WCSession.default.updateApplicationContext(["ipAddress": socketIPField])
            }
            catch {
                print("Error sending IP address: \(error.localizedDescription)")
                print(error)
            }
        } else {
            print("WCSession is not reachable.")
        }
    }
    
    func startWatchMotion() {
        watchCnt = 0
        // self.watchStatusLabel = "\(self.watchCnt) data / \(round(self.watchMeasuredFrequency! * 100) / 100) [Hz]"
        if (WCSession.default.isReachable) {
            do {
                try WCSession.default.updateApplicationContext(["command": "start"])
            }
            catch {
                print(error)
            }
        } else {
            watchStatusLabel = "Not reachable..."
        }
    }
    
    func stopWatchMotion() {
        watchStatusLabel = "Not recording..."
        if (WCSession.default.isReachable) {
            do {
                try WCSession.default.updateApplicationContext(["command": "stop"])
            }
            catch {
                print(error)
            }
        } else {
            watchStatusLabel = "Not reachable..."
        }
    }
    
    func startPhoneMotion() {
        phoneMotionManager = CMMotionManager()
        phoneMotionManager.deviceMotionUpdateInterval = 1.0 / localSamplingRate
        if phoneMotionManager.isDeviceMotionAvailable {
            phoneCnt = 0
            phonePrevTime = NSDate().timeIntervalSince1970
            DispatchQueue.main.async {
                self.phoneStatusLabel = "Started!"
            }
            phoneMotionManager.startDeviceMotionUpdates(using: .xTrueNorthZVertical, to: phoneQueue) { (motion, error) in
                if let motion = motion {
                    let currentTime = NSDate().timeIntervalSince1970
      
                    sensorDataManager.update(deviceID: 0, motion: motion, timestamps: (currentTime, motion.timestamp))
                    sensorDataManager.update(deviceID: 1, motion: motion, timestamps: (currentTime, motion.timestamp))
                    sensorDataManager.update(deviceID: 2, motion: motion, timestamps: (currentTime, motion.timestamp))
                    sensorDataManager.update(deviceID: 3, motion: motion, timestamps: (currentTime, motion.timestamp))
                    sensorDataManager.update(deviceID: 4, motion: motion, timestamps: (currentTime, motion.timestamp))
                    
//                    if let socketClient = self.socketClient {
//                        if socketClient.connection.state == .ready {
//                            let text = "phone:\(currentTime) \(motion.timestamp) \(motion.userAcceleration.x) \(motion.userAcceleration.y) \(motion.userAcceleration.z) \(motion.attitude.quaternion.x) \(motion.attitude.quaternion.y) \(motion.attitude.quaternion.z) \(motion.attitude.quaternion.w) \(motion.attitude.roll) \(motion.attitude.pitch) \(motion.attitude.yaw)\n"
//                            socketClient.send(text: text)
//                        }
//                    }
                    
                    self.phoneCnt += 1
                    if self.phoneCnt % self.nToMeasureFrequency == 0 {
                        let timeDiff = (currentTime - self.phonePrevTime) as Double
                        self.phonePrevTime = currentTime
                        self.phoneMeasuredFrequency = 1.0 / timeDiff * Double(self.nToMeasureFrequency)
                        DispatchQueue.main.async {
                            self.phoneStatusLabel = "\(self.phoneCnt) data / \(round(self.phoneMeasuredFrequency! * 100) / 100) [Hz]"
                        }
                    }
                } else {
                    print(error as Any)
                }
            }
        }
    }
    
    func stopPhoneMotion() {
        self.phoneStatusLabel = "Not Recording..."
        phoneMotionManager.stopDeviceMotionUpdates()
    }
    
    
    func startHeadphoneMotion() {
        headphoneMotionManager = CMHeadphoneMotionManager()
        if headphoneMotionManager.isDeviceMotionAvailable {
            headphoneCnt = 0
            headphonePrevTime = NSDate().timeIntervalSince1970
            DispatchQueue.main.async {
                self.headphoneStatusLabel = "Started!"
            }
            headphoneMotionManager.startDeviceMotionUpdates(to: headphoneQueue) { (motion, error) in
                if let motion = motion {
                    let currentTime = NSDate().timeIntervalSince1970
                    if let socketClient = self.socketClient, socketClient.connection.state == .ready {
                        let text = "headphone:\(currentTime) \(motion.timestamp) \(motion.userAcceleration.x) \(motion.userAcceleration.y) \(motion.userAcceleration.z) \(motion.attitude.quaternion.x) \(motion.attitude.quaternion.y) \(motion.attitude.quaternion.z) \(motion.attitude.quaternion.w)\n"
                        socketClient.send(text: text)
                    }
                    self.headphoneCnt += 1
                    if self.headphoneCnt % self.nToMeasureFrequency == 0 {
                        let timeDiff = currentTime - self.headphonePrevTime
                        self.headphonePrevTime = currentTime
                        self.headphoneMeasuredFrequency = 1.0 / timeDiff * Double(self.nToMeasureFrequency)
                        DispatchQueue.main.async {
                            self.headphoneStatusLabel = "\(self.headphoneCnt) data / \(round(self.headphoneMeasuredFrequency! * 100) / 100) [Hz]"
                        }
                    }
                } else {
                    print("Something went wrong with headphone IMU")
                    if let error = error { print(error) }
                }
            }
        } else {
            DispatchQueue.main.async {
                self.headphoneStatusLabel = "Headphone motion not available"
            }
        }
    }
    
    func stopHeadphoneMotion() {
        self.headphoneStatusLabel = "Not Recording..."
        headphoneMotionManager.stopDeviceMotionUpdates()
    }
    
    func sendSamplingRateToWatch() {
        guard WCSession.default.isReachable else {
            print("Watch is not reachable")
            sessionManager.watchStatusLabel = "Watch not reachable"
            return
        }
        
        let context: [String: Any] = ["samplingRate": localSamplingRate]
        
        do {
            try WCSession.default.updateApplicationContext(context)
            print("Sent sampling rate: \(localSamplingRate) Hz")
        } catch {
            print("Error sending sampling rate: \(error.localizedDescription)")
        }
    }
}

class SessionManager: NSObject, ObservableObject, WCSessionDelegate {
    @Published var watchData: CMDeviceMotion?
    @Published var watchStatusLabel: String = "Not connected"
    @Published var watchCnt: Int = 0
    @Published var connectionError: String?

    let nToMeasureFrequency = 100
    var watchPrevTime: TimeInterval = NSDate().timeIntervalSince1970
    var watchSetFrequency: Double? = 25.0
    var watchMeasuredFrequency: Double? = 25.0
    var socketClient: SocketClient?
    var sensorDataManager: SensorDataManager?
    
    init(socketClient: SocketClient?, sensorDataManager: SensorDataManager?) {
        super.init()
        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
        self.socketClient = socketClient
        self.sensorDataManager = sensorDataManager
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
    }
    
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        if let e = error {
            print("Completed activation with error: \(e.localizedDescription)")
        } else {
            print("Completed activation!")
        }
    }
    
    func session(_ session: WCSession, didReceive file: WCSessionFile) {
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any] = [:]) {
        if let motionData = message["motionData"] as? String {
            //self.watchData = motionData
            if let socketClient = self.socketClient {
                if socketClient.connection.state == .ready {
                    print("receiving watch data")
                    let text = "watch:" + motionData
                    sensorDataManager?.update(dataString: text)
//                    socketClient.send(text: text)
                }
            }
//            watchCnt += 1
//            if watchCnt % nToMeasureFrequency == 0 {
//                let currentTime = NSDate().timeIntervalSince1970
//                let timeDiff = (currentTime - self.watchPrevTime) as Double
//                watchPrevTime = currentTime
//                watchMeasuredFrequency = 1.0 / timeDiff * Double(nToMeasureFrequency)
//                DispatchQueue.main.async {
//                    self.watchStatusLabel = "\(self.watchCnt) data / \(round(self.watchMeasuredFrequency! * 100) / 100) [Hz]"
//                }
//            }
        }
    }
    
    // Handle incoming application contexts
    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String : Any]) {
        DispatchQueue.main.async {
            if let frequency = applicationContext["samplingRate"] as? Double {
                self.watchSetFrequency = frequency
                // Update the sampling rate in your MotionManager or relevant component
                // For example:
                // MotionManager.shared.setSamplingRate(value: frequency)
                print("Received Sampling Rate: \(frequency) Hz")
            }
            
            // Handle other keys if necessary
        }
    }
    
    // Handle incoming messages
    func session(_ session: WCSession, didReceiveMessage message: [String : Any], replyHandler: @escaping ([String : Any]) -> Void) {
        if let request = message["requestSamplingRate"] as? Bool, request == true {
            replyHandler(["samplingRate": self.watchSetFrequency ?? 25.0])
        }
        
        // Handle other messages
    }
    
    // Add a method to send sampling rate to the Watch
    func sendSamplingRate(_ rate: Double) {
        guard WCSession.default.isReachable else {
            print("Watch is not reachable")
            DispatchQueue.main.async {
                self.connectionError = "Watch is not reachable"
            }
            return
        }
        
        let context: [String: Any] = ["samplingRate": rate]
        
        do {
            try WCSession.default.updateApplicationContext(context)
            print("Sent sampling rate: \(rate) Hz")
            UserDefaults.standard.set(rate, forKey: "samplingRate")
        } catch {
            print("Error sending sampling rate: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.connectionError = "Failed to send sampling rate"
            }
        }
    }
}

#Preview {
    ContentView()
}
