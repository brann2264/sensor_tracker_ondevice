import Foundation
import UnityFramework

class UnityManager: NSObject {
    private static var instance: UnityManager!
    private var ufw: UnityFramework!
    private var hostMainWindow: UIWindow!
    
    static func getInstance() -> UnityManager {
        if instance == nil {
            instance = UnityManager()
        }
        return instance
    }
    
    func show() {
        if ufw == nil {
            initUnity()
        }
        ufw.showUnityWindow()
    }
    
    func hide() {
        if ufw != nil {
            ufw.pause(true)
        }
    }
    
    func sendMotionData(_ data: String) {
        if ufw != nil {
            ufw.sendMessageToGO(withName: "DataReceiver",
                               functionName: "ReceiveMotionData",
                               message: data)
        }
    }
    
    private func initUnity() {
        ufw = loadUnityFramework()
        ufw?.setDataBundleId("com.unity3d.framework")
        ufw?.runEmbedded(withArgc: CommandLine.argc,
                        argv: CommandLine.unsafeArgv,
                        appLaunchOpts: nil)
    }
    
    private func loadUnityFramework() -> UnityFramework? {
        let bundlePath = Bundle.main.bundlePath + "/Frameworks/UnityFramework.framework"
        let bundle = Bundle(path: bundlePath)
        
        if bundle?.isLoaded == false {
            bundle?.load()
        }
        
        return bundle?.principalClass?.getInstance()
    }
}
