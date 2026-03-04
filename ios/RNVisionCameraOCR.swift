import Foundation
import VisionCamera
import MLKitVision
import MLKitTextRecognition
import MLKitTextRecognitionChinese
import MLKitTextRecognitionDevanagari
import MLKitTextRecognitionJapanese
import MLKitTextRecognitionKorean
import MLKitCommon
import CoreImage
import UIKit

@objc(RNVisionCameraOCR)
public class RNVisionCameraOCR: FrameProcessorPlugin {

    private var textRecognizer = TextRecognizer()
    private var scanRegion: [String: Int]? = nil
    private static let latinOptions = TextRecognizerOptions()
    private static let chineseOptions = ChineseTextRecognizerOptions()
    private static let devanagariOptions = DevanagariTextRecognizerOptions()
    private static let japaneseOptions = JapaneseTextRecognizerOptions()
    private static let koreanOptions = KoreanTextRecognizerOptions()
    private var data: [String: Any] = [:]
    
    // Performance optimization: configurable frame skipping
    private var frameSkipCount = 0
    private var frameSkipThreshold: Int = 10
    private var isProcessing = false
    
    // Short-term caching for performance
    private var lastProcessedText = ""
    private var lastProcessedTime: TimeInterval = 0
    private var cachedResult: [String: Any]? = nil
    private let cacheTimeoutMs: TimeInterval = 0.150 // Cache results for 150ms

    public override init(proxy: VisionCameraProxyHolder, options: [AnyHashable: Any]! = [:]) {
        super.init(proxy: proxy, options: options)
        scanRegion = options["scanRegion"] as? [String: Int]
        frameSkipThreshold = (options["frameSkipThreshold"] as? Int) ?? 10
        let language = options["language"] as? String ?? "latin"
        switch language {
        case "chinese":
            self.textRecognizer = TextRecognizer.textRecognizer(options: RNVisionCameraOCR.chineseOptions)
        case "devanagari":
            self.textRecognizer = TextRecognizer.textRecognizer(options: RNVisionCameraOCR.devanagariOptions)
        case "japanese":
            self.textRecognizer = TextRecognizer.textRecognizer(options: RNVisionCameraOCR.japaneseOptions)
        case "korean":
            self.textRecognizer = TextRecognizer.textRecognizer(options: RNVisionCameraOCR.koreanOptions)
        default:
            self.textRecognizer = TextRecognizer.textRecognizer(options: RNVisionCameraOCR.latinOptions)
        }
    }


    public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable: Any]?) -> Any {
        // Performance optimization: skip frames to reduce processing load
        frameSkipCount += 1
        if frameSkipCount < frameSkipThreshold || isProcessing {
            return getCachedResult()
        }
        frameSkipCount = 0
        isProcessing = true
        
        defer {
            isProcessing = false
        }
        
        let buffer = frame.buffer
        var image: VisionImage?

        do {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(buffer) else {
                return [:]
            }
            
            // Convert pixel buffer to CIImage and apply correct orientation
            // This ensures the image is physically rotated, not just metadata
            let baseCIImage = CIImage(cvPixelBuffer: pixelBuffer)
            let orientedCIImage = baseCIImage.oriented(getCIImageOrientation(from: frame.orientation))
            
            let context = CIContext(options: nil)
            guard let cgImage = context.createCGImage(orientedCIImage, from: orientedCIImage.extent) else {
                return [:]
            }
            
            if scanRegion != nil {
                // Apply scan region cropping
                let imgWidth = Double(cgImage.width)
                let imgHeight = Double(cgImage.height)
                let left: Double = Double(scanRegion?["left"] ?? 0) / 100.0 * imgWidth
                let top: Double = Double(scanRegion?["top"] ?? 0) / 100.0 * imgHeight
                let width: Double = Double(scanRegion?["width"] ?? 100) / 100.0 * imgWidth
                let height: Double = Double(scanRegion?["height"] ?? 100) / 100.0 * imgHeight
                let cropRegion = CGRect(
                    x: left,
                    y: top,
                    width: width,
                    height: height
                )
                guard let croppedCGImage = cgImage.cropping(to: cropRegion) else {
                    return [:]
                }
                let uiImage = UIImage(cgImage: croppedCGImage, scale: 1.0, orientation: .up)
                image = VisionImage(image: uiImage)
            } else {
                // Use full image with correct orientation already applied
                let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
                image = VisionImage(image: uiImage)
            }
            
            let result = try self.textRecognizer.results(in: image!)
            let blocks = RNVisionCameraOCR.processBlocks(blocks: result.blocks)
            data["resultText"] = result.text
            data["blocks"] = blocks
            
            let resultText = result.text
            let currentTime = Date().timeIntervalSince1970
            
            if result.text.isEmpty {
                updateCache(text: "", time: currentTime, result: nil)
                return [:]
            }else{
                updateCache(text: resultText, time: currentTime, result: data)
                return data
            }
        } catch {
            print("Failed to recognize text: \(error.localizedDescription).")
            return getCachedResult()
        }
    }

    private func updateCache(text: String, time: TimeInterval, result: [String: Any]?) {
        lastProcessedText = text
        lastProcessedTime = time
        cachedResult = result
    }
    
    private func getCachedResult() -> [String: Any] {
        let currentTime = Date().timeIntervalSince1970
        if currentTime - lastProcessedTime < cacheTimeoutMs,
           let cachedResult {
            return cachedResult
        }
        return [:]
    }

      static func processBlocks(blocks:[TextBlock]) -> Array<Any> {
        var blocksArray : [Any] = []
        for block in blocks {
            var blockData : [String:Any] = [:]
            blockData["blockText"] = block.text
            blockData["blockCornerPoints"] = processCornerPoints(block.cornerPoints)
            blockData["blockFrame"] = processFrame(block.frame)
            blockData["lines"] = processLines(lines: block.lines)
            blocksArray.append(blockData)
        }
        return blocksArray
    }

    private static func processLines(lines:[TextLine]) -> Array<Any> {
        var linesArray : [Any] = []
        for line in lines {
            var lineData : [String:Any] = [:]
            lineData["lineText"] = line.text
            lineData["lineLanguages"] = processRecognizedLanguages(line.recognizedLanguages)
            lineData["lineCornerPoints"] = processCornerPoints(line.cornerPoints)
            lineData["lineFrame"] = processFrame(line.frame)
            lineData["elements"] = processElements(elements: line.elements)
            linesArray.append(lineData)
        }
        return linesArray
    }

    private static func processElements(elements:[TextElement]) -> Array<Any> {
        var elementsArray : [Any] = []

        for element in elements {
            var elementData : [String:Any] = [:]
              elementData["elementText"] = element.text
              elementData["elementCornerPoints"] = processCornerPoints(element.cornerPoints)
              elementData["elementFrame"] = processFrame(element.frame)

            elementsArray.append(elementData)
          }

        return elementsArray
    }

    private static func processRecognizedLanguages(_ languages: [TextRecognizedLanguage]) -> [String] {

            var languageArray: [String] = []

            for language in languages {
                guard let code = language.languageCode else {
                    print("No language code exists")
                    break;
                }
                if code.isEmpty{
                    languageArray.append("und")
                }else {
                    languageArray.append(code)

                }
            }

            return languageArray
        }

    private static func processCornerPoints(_ cornerPoints: [NSValue]) -> [[String: CGFloat]] {
        return cornerPoints.compactMap { $0.cgPointValue }.map { ["x": $0.x, "y": $0.y] }
    }

    private static func processFrame(_ frameRect: CGRect) -> [String: CGFloat] {
        let offsetX = (frameRect.midX - ceil(frameRect.width)) / 2.0
        let offsetY = (frameRect.midY - ceil(frameRect.height)) / 2.0

        let x = frameRect.maxX + offsetX
        let y = frameRect.minY + offsetY

        return [
            "x": frameRect.midX + (frameRect.midX - x),
            "y": frameRect.midY + (y - frameRect.midY),
            "width": frameRect.width,
            "height": frameRect.height,
            "boundingCenterX": frameRect.midX,
            "boundingCenterY": frameRect.midY
    ]
    }

    /// Converts UIImage.Orientation from VisionCamera frame to CGImagePropertyOrientation
    /// This ensures the CIImage is physically rotated to match the device orientation.
    /// iOS camera sensors are mounted in landscape, so we need to rotate the pixel data
    /// to match portrait mode when the device is held upright.
    private func getCIImageOrientation(from uiOrientation: UIImage.Orientation) -> CGImagePropertyOrientation {
        switch uiOrientation {
        case .up:
            // Device is in portrait, camera sensor is landscape (90° rotated)
            // So we need to rotate 90° clockwise to get upright text
            return .right
        case .down:
            // Device is upside down, rotate 270° clockwise (or 90° counter-clockwise)
            return .left
        case .left:
            // Device rotated 90° counter-clockwise (landscape left)
            // Camera sensor is already in landscape, so no rotation needed
            return .up
        case .right:
            // Device rotated 90° clockwise (landscape right)
            // Camera sensor is already in landscape, rotate 180°
            return .down
        case .upMirrored:
            return .rightMirrored
        case .downMirrored:
            return .leftMirrored
        case .leftMirrored:
            return .upMirrored
        case .rightMirrored:
            return .downMirrored
        @unknown default:
            // Default to portrait orientation (most common use case)
            return .right
        }
    }
}
