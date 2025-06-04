//
//  Untitled.swift
//  SensorTracker
//
//  Created by Brian Chen on 5/2/25.
//
import CoreML
import Foundation
import simd
import Collections
import Spatial
import Network
import CoreMotion

struct Constants {
    static let DEVICE_IDS: [String: Int] = [
        "Left_phone": 0,
        "Left_watch": 1,
        "Left_headphone": 2,
        "Right_phone": 3,
        "Right_watch": 4
    ]
    
    static let KEYS = ["unix_timestamp", "sensor_timestamp", "accel_x", "accel_y", "accel_z", "quart_x", "quart_y", "quart_z", "quart_w", "roll", "pitch", "yaw"]
    static let STOP = "stop"
    static let SEP = ":"
    static let combos: [String: [Int]] = [
        "lw_rp_h": [0, 3, 4],
        "rw_rp_h": [1, 3, 4],
        "lw_lp_h": [0, 2, 4],
        "rw_lp_h": [1, 2, 4],
        "lw_lp":   [0, 2],
        "lw_rp":   [0, 3],
        "rw_lp":   [1, 2],
        "rw_rp":   [1, 3],
        "lp_h":    [2, 4],
        "rp_h":    [3, 4],
        "lp":      [2],
        "rp":      [3]
    ]
}

class MobilePoserManager {
    
    // class variables here
    private var model: MobilePoserComplete
    private var modelInitial: MobilePoserCompleteInitial
    private var calibrator: Calibrator
    
//    private var processInitial: ProcessInputsInitial
//    private var processRegular: ProcessInputs
    
    private var h: MLMultiArray
    private var c: MLMultiArray
    var lastLFootPos: SIMD3<Float> = SIMD3<Float>(0.1283, -0.9559,  0.0750)
    var lastRFootPos: SIMD3<Float> =  SIMD3<Float>(-0.1194, -0.9564,  0.0774)
    var gravityVelocity: SIMD3<Float> = SIMD3<Float>(0, -0.018, 0)
    var currentRootY: Float = 0
    var floorY: Float = -0.9563523530960083
    var lastRootPos: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    var probThreshold: [Float] = [0.5, 0.9]
    var imuHistory: MLMultiArray? = nil
    
    
    // initializer
    init(){
        
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try! MobilePoserComplete(configuration: config)
        self.modelInitial = try! MobilePoserCompleteInitial(configuration: MLModelConfiguration())
        self.calibrator = try! Calibrator(configuration: MLModelConfiguration())

        do {
            let shape: [NSNumber] = [2, 1, 256].map { NSNumber(value: $0) }
            self.h = try MLMultiArray(shape: shape, dataType: .float32)
            self.c = try MLMultiArray(shape: shape, dataType: .float32)
            let count = shape.reduce(1) { $0 * $1.intValue }    // total number of elements
            let byteCount = count * MemoryLayout<Float32>.size   // total bytes
            memset(self.h.dataPointer, 0, byteCount)
            memset(self.c.dataPointer, 0, byteCount)
        } catch {
            fatalError("Could not allocate array: \(error)")
        }
    }
    
    func probToWeight(_ p: Float) -> Float {
        let lower = probThreshold[0]
        let upper = probThreshold[1]
        let clamped = min(max(p, lower), upper)
        return (clamped - lower) / (upper - lower)
    }
    
    // class methods
    func predict(ori_raw: MLMultiArray,
                 acc_raw: MLMultiArray,
                 acc_offsets: MLMultiArray,
                 smpl2imu: MLMultiArray,
                 device2bone: MLMultiArray) ->(pose: MLMultiArray,
                                              joints: MLMultiArray,
                                              rootPos: MLMultiArray,
                                              contact: MLMultiArray)? {
        var curr_pose : MLMultiArray
        var pred_joints : MLMultiArray
        var pred_vel : MLMultiArray
        var contact : MLMultiArray
        let startTime = DispatchTime.now()
        if self.imuHistory == nil {
            guard let output = try? self.modelInitial.prediction(ori_raw_1: ori_raw,
                                                          acc_raw: acc_raw,
                                                          acc_offsets: acc_offsets,
                                                          smpl2imu: smpl2imu,
                                                          device2bone: device2bone,
                                                          h: self.h,
                                                          c_1: self.c)
            else {
                return nil
            }
            curr_pose = output.var_632
            pred_joints = output.var_641
            pred_vel = output.var_665
            contact = output.var_647
            self.h = output.var_434
            self.c = output.var_435
            self.imuHistory = output.imu
            
        } else {
            guard let output = try? self.model.prediction(imu_1: self.imuHistory!,
                                                          ori_raw_1: ori_raw,
                                                          acc_raw: acc_raw,
                                                          acc_offsets: acc_offsets,
                                                          smpl2imu: smpl2imu,
                                                          device2bone: device2bone,
                                                          h: self.h,
                                                          c_1: self.c)
            else {
                return nil
            }
            curr_pose = output.var_641
            pred_joints = output.var_650
            pred_vel = output.var_674
            contact = output.var_656
            self.h = output.var_443
            self.c = output.var_444
            self.imuHistory = output.imu
        }
        
        let endTime = DispatchTime.now()
        let elapsedTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds // Time in nanoseconds (UInt64)
        let durationInSeconds = Double(elapsedTime) / 1_000_000_000.0 // Convert to seconds
        
        print("----------------------------------------------------")
        print("Model Execution time:")
        print("Nanoseconds: \(elapsedTime)")
        print("Seconds: \(String(format: "%.6f", durationInSeconds)) seconds") // Formatted to 6 decimal places
        
        let lfootPos = vector3_fromArray(from: pred_joints, at: 10)
        let rfootPos = vector3_fromArray(from: pred_joints, at: 11)
        let predVel = vector3_fromArray(from: pred_vel, at: 0)
        let contactX = contact[0].floatValue
        let contactY = contact[1].floatValue
        let contactVel: SIMD3<Float> = contactX > contactY
          ? (lastLFootPos - lfootPos + gravityVelocity)
          : (lastRFootPos - rfootPos + gravityVelocity)

        let w = probToWeight(max(contactX, contactY))

        // linear interp for velcoity
        var velocity = lerp(predVel, contactVel, t: w)

        let currentFootY = currentRootY + min(lfootPos.y, rfootPos.y)
        if currentFootY + velocity.y <= floorY {
          velocity.y = floorY - currentFootY
        }
        
        // update states
        currentRootY += velocity.y
        lastLFootPos = lfootPos
        lastRFootPos = rfootPos
        lastRootPos += velocity
        let rootPos = try! mlArray_fromVector3(from: lastRootPos)
        
        return (curr_pose, pred_joints, rootPos, contact)
    }
}




func mlArray_fromVector3(from vector: SIMD3<Float>) throws -> MLMultiArray {
    // converts SIMD3 into MLMultiArray
    let array = try MLMultiArray(shape: [3], dataType: .float32)
    
    array[0] = NSNumber(value: vector.x)
    array[1] = NSNumber(value: vector.y)
    array[2] = NSNumber(value: vector.z)
    
    return array
}

func vector3_fromArray(from multiArray: MLMultiArray, at jointIndex: Int) -> SIMD3<Float> {
    // indexes a SIMD3 from a MLMultiArray
    let shape = multiArray.shape.map { $0.intValue }
        
    switch shape.count {
    case 1 where shape[0] == 3:
        let x = multiArray[0].floatValue
        let y = multiArray[1].floatValue
        let z = multiArray[2].floatValue
        return SIMD3<Float>(x, y, z)
        
    case 2 where shape[1] == 3:
        let strides = multiArray.strides.map { $0.intValue }
        let rowStride  = strides[0]
        let compStride = strides[1]
        
        let base = jointIndex * rowStride
        let x = multiArray[base + 0 * compStride].floatValue
        let y = multiArray[base + 1 * compStride].floatValue
        let z = multiArray[base + 2 * compStride].floatValue
        return SIMD3<Float>(x, y, z)
        
    default:
        preconditionFailure("Expected shape [3] or [N, 3], got \(shape)")
    }
}

func lerp(_ a: SIMD3<Float>, _ b: SIMD3<Float>, t: Float) -> SIMD3<Float> {
    // linear interp
    return a * (1 - t) + b * t
}

class StreamClient: ObservableObject {
    let connection: NWConnection
    private var buffer = Data()
    private var mlManager: MobilePoserManager
    @Published var isConnected: Bool = false

    init(host: String, port: UInt16) {
        connection = NWConnection(
            host: .init(host),
            port: .init(rawValue: port)!,
            using: .tcp
        )
        mlManager = MobilePoserManager()
    }

    func start() {
        connection.stateUpdateHandler = { [weak self] newState in
            DispatchQueue.main.async {
                switch newState {
                case .ready:
                    self?.isConnected = true
                    print("Connected to server")
                    self?.receiveLoop()
                case .failed(let error):
                    self?.isConnected = false
                    print("Connection failed:", error)
                default:
                    break
                }
            }
        }

        connection.start(queue: .global(qos: .background))
    }


    func stop() {
        connection.cancel()
        DispatchQueue.main.async {
            self.isConnected = false
        }
    }

    private func receiveLoop() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { data, _, isComplete, error in
            if let data = data, !data.isEmpty {
                debugPrint("Received tensors")
                self.buffer.append(data)
                self.processBuffer()
            }
            if isComplete || error != nil {
                print("Connection closed or error:", error as Any)
                return
            }
            self.receiveLoop()
        }
    }

    private func processBuffer() {
        
        // needs at least 4 bytes to read how long the whole package is
        while buffer.count >= 4 {

            // check if we have complete package
            guard buffer.count >= 4 else { return }

            // read length
            let totalLen = Int(buffer.prefix(4)
                                 .withUnsafeBytes { $0.load(as: UInt32.self) }
                                 .littleEndian)

            // if we have everything
            guard buffer.count >= 4 + totalLen else { return }

            // extract and remove
            let payload = buffer.dropFirst(4).prefix(totalLen)
            buffer.removeFirst(4 + totalLen)

            // parse and run model
            if let arrays = parseTensors(from: payload) {
                
                guard let output = mlManager.predict(ori_raw: arrays[0],
                                                     acc_raw: arrays[1],
                                                     acc_offsets: arrays[2],
                                                     smpl2imu: arrays[3],
                                                     device2bone: arrays[4]) else {
                    debugPrint("model failed")
                    continue
                }
                sendOutputs(outputs: output)
            }
        }
    }
    
    private func sendOutputs(outputs: (pose: MLMultiArray,
                                       joints: MLMultiArray,
                                       rootPos: MLMultiArray,
                                       contact: MLMultiArray)){

        let arrays = [outputs.pose, outputs.joints, outputs.rootPos, outputs.contact]
        var body = Data()

        // tensor count
        var cnt = UInt32(arrays.count).littleEndian
        withUnsafeBytes(of: &cnt) { body.append(contentsOf: $0) }

        for ml in arrays {
            // dims size
            var rank = UInt32(ml.shape.count).littleEndian
            withUnsafeBytes(of: &rank) { body.append(contentsOf: $0) }

            // dims
            for num in ml.shape {
                var d = UInt32(truncating: num).littleEndian
                withUnsafeBytes(of: &d) { body.append(contentsOf: $0) }
            }

            // count of this tensor
            let elementCount = ml.count
            var byteCount = UInt32(elementCount * MemoryLayout<Float32>.size).littleEndian
            withUnsafeBytes(of: &byteCount) { body.append(contentsOf: $0) }

            // tensor data
            let rawPtr = ml.dataPointer.assumingMemoryBound(to: UInt8.self)
            body.append(rawPtr, count: Int(byteCount.littleEndian))
        }

        // total length in front
        var total = UInt32(body.count).littleEndian
        var packet = Data()
        withUnsafeBytes(of: &total) { packet.append(contentsOf: $0) }
        packet.append(body)

        connection.send(content: packet, completion: .contentProcessed { error in
            if let e = error { print("sendOutputs error:", e) }
        })
    }

    private func parseTensors(from body: Data) -> [MLMultiArray]? {
        var offset = 0

        func readUInt32() -> UInt32 {
            let v = body.withUnsafeBytes { ptr in
                ptr.load(fromByteOffset: offset, as: UInt32.self)
            }
            offset += 4
            return v.littleEndian
        }

        // tensor count
        let count = Int(readUInt32())

        var results: [MLMultiArray] = []

        for _ in 0..<count {
            // rank
            let rank = Int(readUInt32())

            // dims
            var shape: [NSNumber] = []
            for _ in 0..<rank {
                shape.append(NSNumber(value: Int(readUInt32())))
            }

            // payload length
            let byteCount32 = readUInt32()
            let byteCount = Int(byteCount32)

            // make sure there are enough bytes left
            guard body.count >= offset + byteCount else {
                debugPrint("Not enough bytes left for entire array")
                return nil }

            // extract payload
            let payload = body.dropFirst(offset).prefix(byteCount)
            offset += byteCount

            // build MLMultiArray
            do {
                let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
                let _ = payload.withUnsafeBytes { ptr in
                    memcpy(mlArray.dataPointer, ptr.baseAddress!, byteCount)
                }
                results.append(mlArray)
            } catch {
                print("Failed to allocate MLMultiArray:", error)
                return nil
            }
        }

        return results
    }

}



//
// Adapted from sensor_utils.py
//

class SensorDataManager {

    private var buffer_size = 45

    typealias DeviceID = Int
    
    private(set) var rawAccBuffer   = Dictionary<DeviceID, Deque<SIMD3<Double>>>()
    private(set) var rawOriBuffer   = Dictionary<DeviceID, Deque<SIMD4<Double>>>()
    private(set) var referenceTimes = Dictionary<DeviceID, (Double, Double)>()
    
    init() {
        for id in Constants.DEVICE_IDS.values {
            rawAccBuffer[id]       = []
            rawOriBuffer[id]       = []
            referenceTimes[id]     = (-1, -1)
        }
    }
        

    func update(
        deviceID: DeviceID,
        motion: CMDeviceMotion,
        timestamps: (Double, Double)
    ) -> Double {
        
        if referenceTimes[deviceID] == nil {
            referenceTimes[deviceID] = timestamps
        }
        
        guard let ref = referenceTimes[deviceID] else {
            return timestamps.1
        }
        
        let currTimestamp = ref.0 + (timestamps.1 - ref.1)
        
        let currAcc = SIMD3<Double>(motion.userAcceleration.x, motion.userAcceleration.y, motion.userAcceleration.z)
        let currOri = SIMD4<Double>(motion.attitude.quaternion.x, motion.attitude.quaternion.y, motion.attitude.quaternion.z, motion.attitude.quaternion.w)
        
        rawAccBuffer[deviceID]?.append(currAcc)
        if let count = rawAccBuffer[deviceID]?.count, count > buffer_size {
            rawAccBuffer[deviceID]?.removeFirst()
        }
        
        rawOriBuffer[deviceID]?.append(currOri)
        if let count = rawOriBuffer[deviceID]?.count, count > buffer_size {
            rawOriBuffer[deviceID]?.removeFirst()
        }
        
        // Update last
        referenceTimes[deviceID] = (ref.0, timestamps.1)
        return currTimestamp
    }
    
    func getTimestamp(deviceID: DeviceID) -> Double? {
        return referenceTimes[deviceID]?.1
    }
    func getOrientation(deviceID: DeviceID) -> SIMD4<Double>? {
        return rawOriBuffer[deviceID]?.last
    }
    func getAcceleration(deviceID: DeviceID) -> SIMD3<Double>? {
        return rawAccBuffer[deviceID]?.last
    }
    
    func getCurrentBuffer() -> (Dictionary<DeviceID, Deque<SIMD3<Double>>>, Dictionary<DeviceID, Deque<SIMD4<Double>>>){
        return (rawAccBuffer, rawOriBuffer)
    }
    
    func getMeanMeasurement(numSeconds: TimeInterval = 3.0,
                                bufferLen: Int = 120) -> (meanQuaternions: Dictionary<DeviceID, SIMD4<Double>>,
                                                          meanAccelerations: Dictionary<DeviceID, SIMD3<Double>>) {
        let oldBufferLen = self.buffer_size
        self.buffer_size = bufferLen
        
        Thread.sleep(forTimeInterval: numSeconds)
        
        let (accBuffer, oriBuffer) = getCurrentBuffer()
        
        self.buffer_size = oldBufferLen
        
        var accMeans = Dictionary<DeviceID, SIMD3<Double>>()
        var oriMeans = Dictionary<DeviceID, SIMD4<Double>>()
        
        for id in Constants.DEVICE_IDS.values {
            
            var accSum = SIMD3<Double>()
            var oriSum = SIMD4<Double>()
            var accCount = 0
            var oriCount = 0
            
            let accD = accBuffer[id]!
            let oriD = oriBuffer[id]!
            
            for acc in accD{
                accSum += acc
                accCount += 1
            }
            
            for ori in oriD{
                oriSum += ori
                oriCount += 1
            }
            
            accMeans[id] = accSum / Double(accCount)
            oriMeans[id] = oriSum / Double(oriCount)
        }
        
        return (oriMeans, accMeans)
        
    }

}

