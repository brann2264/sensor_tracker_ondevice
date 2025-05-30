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
}

class MobilePoserManager {
    
    // class variables here
    private var model: MobilePoser
    private var h: MLMultiArray
    private var c: MLMultiArray
    var lastLFootPos: SIMD3<Float> = SIMD3<Float>(0.1283, -0.9559,  0.0750)
    var lastRFootPos: SIMD3<Float> =  SIMD3<Float>(-0.1194, -0.9564,  0.0774)
    var gravityVelocity: SIMD3<Float> = SIMD3<Float>(0, -0.018, 0)
    var currentRootY: Float = 0
    var floorY: Float = -0.9563523530960083
    var lastRootPos: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    var probThreshold: [Float] = [0.5, 0.9]
    
    // initializer
    init(){
        
        let config = MLModelConfiguration()
        self.model = try! MobilePoser(configuration: config)
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
    func predict(imu_input: MLMultiArray, input_length: MLMultiArray) -> (pose: MLMultiArray,
                                                                          joints: MLMultiArray,
                                                                          vel: MLMultiArray,
                                                                          contact: MLMultiArray)? {
        guard let output = try? self.model.prediction(batch: imu_input, h: self.h, c: self.c, input_lengths: input_length) else {
            return nil
        }
        
        let pred_pose = output.var_290
        let pred_joints = output.var_299
        let pred_vel = output.var_323
        let contact = output.var_305
        self.h = output.var_187
        self.c = output.var_188
        
        let lfootPos = vector3_fromArray(from: pred_joints, at: 10)
        let rfootPos = vector3_fromArray(from: pred_joints, at: 11)
        let predVel = vector3_fromArray(from: pred_vel, at: 0)
        let contactX = contact[0].floatValue
        let contactY = contact[1].floatValue
        
        let contactVel: SIMD3<Float> = contactX > contactY
          ? (lastLFootPos - lfootPos + gravityVelocity)
          : (lastRFootPos - rfootPos + gravityVelocity)

        // compute your blend weight from the max contact probability
        let w = probToWeight(max(contactX, contactY))

        // linear interpolation: predVel + (contactVel – predVel) * w
        var velocity = lerp(predVel, contactVel, t: w)

        // figure out where the “lowest” foot is in Y
        let currentFootY = currentRootY + min(lfootPos.y, rfootPos.y)

        // if applying this velocity would push you below the floor, clamp
        if currentFootY + velocity.y <= floorY {
          velocity.y = floorY - currentFootY
        }

        // finally, update all your state
        currentRootY += velocity.y
        lastLFootPos = lfootPos
        lastRFootPos = rfootPos
        lastRootPos += velocity
        
        let rootPos = try! mlArray_fromVector3(from: lastRootPos)
        
        return (pred_pose, pred_joints, rootPos, contact)
//        return (pred_pose, pred_joints, pred_vel, contact)
    }
    
    
}

func mlArray_fromVector3(from vector: SIMD3<Float>) throws -> MLMultiArray {
    // 1) Allocate a 1-D MLMultiArray of size 3
    let array = try MLMultiArray(shape: [3], dataType: .float32)
    
    // 2) Populate its elements
    array[0] = NSNumber(value: vector.x)
    array[1] = NSNumber(value: vector.y)
    array[2] = NSNumber(value: vector.z)
    
    return array
}

func vector3_fromArray(from multiArray: MLMultiArray, at jointIndex: Int) -> SIMD3<Float> {
    // Validate shape
    let shape = multiArray.shape.map { $0.intValue }
    precondition(shape.count == 2 && shape[1] == 3,
                 "Expected MLMultiArray of shape [N, 3], got \(shape)")
    
    // Compute strides
    let strides = multiArray.strides.map { $0.intValue }
    let rowStride   = strides[0]
    let compStride  = strides[1]
    
    // Base offset for this joint
    let base = jointIndex * rowStride
    
    // Read components
    let x = multiArray[base + 0 * compStride].floatValue
    let y = multiArray[base + 1 * compStride].floatValue
    let z = multiArray[base + 2 * compStride].floatValue
    
    return SIMD3<Float>(x, y, z)
}

func lerp(_ a: SIMD3<Float>, _ b: SIMD3<Float>, t: Float) -> SIMD3<Float> {
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

            // full packet?
            guard buffer.count >= 4 + totalLen else { return }

            // extract and remove
            let payload = buffer.dropFirst(4).prefix(totalLen)
            buffer.removeFirst(4 + totalLen)

            // parse and run modela
            if let arrays = parseTensors(from: payload) {
                print("IMU input Shape:", arrays[0].shape)
                print("Input length Shape:", arrays[1].shape)
                debugPrint("Running model")
                guard let output = mlManager.predict(imu_input: arrays[0], input_length:arrays[1]) else {
                    debugPrint("model failed")
                    continue
                }
                debugPrint("Sending outputs")
                sendOutputs(outputs: output)
            }
        }
    }
    
    private func sendOutputs(outputs: (pose: MLMultiArray,
                                       joints: MLMultiArray,
                                       vel: MLMultiArray,
                                       contact: MLMultiArray)){

        let arrays = [outputs.pose, outputs.joints, outputs.vel, outputs.contact]
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

        // Safe UInt32 reader (little-endian) with bounds check
        func readUInt32() -> UInt32 {
            let v = body.withUnsafeBytes { ptr in
                ptr.load(fromByteOffset: offset, as: UInt32.self)
            }
            offset += 4
            return v.littleEndian
        }

        // 1) Read tensor count
        let count = Int(readUInt32())

        var results: [MLMultiArray] = []

        for _ in 0..<count {
            // 2) Read rank
            let rank = Int(readUInt32())

            // 3) Read dims
            var shape: [NSNumber] = []
            for _ in 0..<rank {
                shape.append(NSNumber(value: Int(readUInt32())))
            }
            debugPrint("Packet says shape:", shape)

            // 4) Read payload length
            let byteCount32 = readUInt32()
            let byteCount = Int(byteCount32)

            // 5) Make sure we have enough bytes left
            guard body.count >= offset + byteCount else {
                debugPrint("Not enough bytes left for entire array")
                return nil }

            // 6) Extract payload safely
            let payload = body.dropFirst(offset).prefix(byteCount)
            offset += byteCount

            // 7) Build the MLMultiArray
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

    private let BUFFER_SIZE = 50

    typealias DeviceID = Int
    typealias Vector3    = SIMD3<Double>
    
    private(set) var rawAccBuffer   = Dictionary<DeviceID, Deque<Vector3>>()
    private(set) var rawOriBuffer   = Dictionary<DeviceID, Deque<simd_quatd>>()
    private(set) var calibrationQuats = Dictionary<DeviceID, simd_quatd>()
    private(set) var virtualAcc     = Dictionary<DeviceID, Vector3>()
    private(set) var virtualOri     = Dictionary<DeviceID, simd_quatd>()
    private(set) var referenceTimes = Dictionary<DeviceID, (Double, Double)>()
    
    init() {
        for id in Constants.DEVICE_IDS.values {
            rawAccBuffer[id]       = []
            rawOriBuffer[id]       = []
            calibrationQuats[id]   = simd_quatd(ix:0, iy:0, iz:0, r:1)
            virtualAcc[id]         = Vector3(0, 0, 0)
            virtualOri[id]         = simd_quatd(ix:0, iy:0, iz:0, r:1)
            referenceTimes[id]     = (-1, -1)
        }
    }
        

    func update(
        deviceID: DeviceID,
        currAcc: Vector3,
        currOri: simd_quatd,
        timestamps: (Double, Double)
    ) -> Double {
        
        if referenceTimes[deviceID] == nil {
            referenceTimes[deviceID] = timestamps
        }
        
        guard let ref = referenceTimes[deviceID] else {
            return timestamps.1
        }
        
        let currTimestamp = ref.0 + (timestamps.1 - ref.1)
        
        // Store acc
        rawAccBuffer[deviceID]?.append(currAcc)
        if let count = rawAccBuffer[deviceID]?.count, count > BUFFER_SIZE {
            rawAccBuffer[deviceID]?.removeFirst()
        }
        
        // Store ori
        rawOriBuffer[deviceID]?.append(currOri)
        if let count = rawOriBuffer[deviceID]?.count, count > BUFFER_SIZE {
            rawOriBuffer[deviceID]?.removeFirst()
        }
        
        // Update last
        referenceTimes[deviceID] = (ref.0, timestamps.1)
        return currTimestamp
    }
    
    /// Compute a mean “zero” quaternion from the last 30 samples
    func calibrate() {
        for (id, oriBuf) in rawOriBuffer {
            guard oriBuf.count >= 30 else {
                print("Not enough data to calibrate for device \(id).")
                continue
            }
            
            let last30 = oriBuf.suffix(30)
            let ref = last30[0]
            // Simple average then normalize:
            var sum = SIMD4<Double>.zero
            for q in last30 {
              let v = q.vector
              // dot(ref, q) < 0 → opposite hemisphere
              sum += (dot(ref.vector, v) < 0) ? -v : v
            }

            calibrationQuats[id] = simd_quatd(vector:simd_normalize(sum))
        }
    }
    
    /// Helpers to read back the latest values
    func getTimestamp(deviceID: DeviceID) -> Double? {
        return referenceTimes[deviceID]?.1
    }
    func getOrientation(deviceID: DeviceID) -> simd_quatd? {
        return rawOriBuffer[deviceID]?.last
    }
    func getAcceleration(deviceID: DeviceID) -> Vector3? {
        return rawAccBuffer[deviceID]?.last
    }
    
    /// Update your “virtual” outputs
    func updateVirtual(deviceID: DeviceID,
                       glbAcc: Vector3,
                       glbOri: simd_quatd)
    {
        virtualAcc[deviceID] = glbAcc
        virtualOri[deviceID] = glbOri
    }
}

func sensor2global(
  ori: simd_quatd,
  acc: SIMD3<Double>,
  calibrationQuats: [Int: simd_quatd],
  deviceID: Int
) -> (globalOri: simd_quatd, globalAcc: SIMD3<Double>) {
  
    guard let deviceMeanQuat = calibrationQuats[deviceID] else {
      fatalError("No calibration quaternion for device \(deviceID)")
    }
    
    let ogMat       = double3x3(ori)
    let globalFrame = double3x3(deviceMeanQuat)
    

    let globalMat = globalFrame.transpose * ogMat
    let globalOri = simd_quatd(globalMat)
    
    let accInBody = ogMat * acc
    let globalAcc = globalFrame.transpose * accInBody
    
    return (globalOri, globalAcc)
}

