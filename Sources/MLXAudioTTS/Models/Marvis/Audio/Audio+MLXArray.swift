import AVFoundation
import Foundation
import MLX

/// Returns (sampleRate, audio).
public func loadAudioArray(from url: URL) throws -> (Double, MLXArray) {
    let file = try AVAudioFile(forReading: url)

    let inFormat = file.processingFormat
    let totalFrames = AVAudioFrameCount(file.length)
    guard let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: totalFrames) else {
        throw NSError(domain: "WAVLoader", code: -1, userInfo: [NSLocalizedDescriptionKey: "Buffer alloc failed"])
    }
    try file.read(into: inBuffer)

    if inFormat.commonFormat == .pcmFormatFloat32, let chans = inBuffer.floatChannelData {
        let frames = Int(inBuffer.frameLength)
        let channels: [[Float]] = (0..<Int(inFormat.channelCount)).map { c in
            let ptr = chans[c]
            return Array(UnsafeBufferPointer(start: ptr, count: frames))
        }
        return (inFormat.sampleRate, MLXArray(channels[0]))
    }

    let floatFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                    sampleRate: inFormat.sampleRate,
                                    channels: inFormat.channelCount,
                                    interleaved: false)
    guard let floatFormat = floatFormat else {
        throw NSError(domain: "WAVLoader", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to create float format"])
    }
    guard let converter = AVAudioConverter(from: inFormat, to: floatFormat) else {
        throw NSError(domain: "WAVLoader", code: -4, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio converter"])
    }
    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: floatFormat, frameCapacity: totalFrames) else {
        throw NSError(domain: "WAVLoader", code: -2, userInfo: [NSLocalizedDescriptionKey: "Out buffer alloc failed"])
    }

    var consumed = false
    var convError: NSError?
    let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
        if consumed {
            outStatus.pointee = .endOfStream
            return nil
        } else {
            consumed = true
            outStatus.pointee = .haveData
            return inBuffer
        }
    }
    converter.convert(to: outBuffer, error: &convError, withInputFrom: inputBlock)
    if let e = convError { throw e }

    let frames = Int(outBuffer.frameLength)
    guard let floatChannelData = outBuffer.floatChannelData else {
        throw NSError(domain: "WAVLoader", code: -5, userInfo: [NSLocalizedDescriptionKey: "Failed to get float channel data"])
    }
    let channels: [[Float]] = (0..<Int(floatFormat.channelCount)).map { c in
        let ptr = floatChannelData[c]
        return Array(UnsafeBufferPointer(start: ptr, count: frames))
    }
    return (floatFormat.sampleRate, MLXArray(channels[0]))
}

public enum WAVWriterError: Error {
    case noFrames
    case bufferAllocFailed
    case fileNotOpen
}

/// Streaming WAV writer that writes audio chunks directly to file
public class StreamingWAVWriter {
    private var audioFile: AVAudioFile?
    private let format: AVAudioFormat
    private let url: URL
    private var totalFramesWritten: Int = 0

    public init(url: URL, sampleRate: Double, channels: Int = 1) throws {
        self.url = url
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: AVAudioChannelCount(channels),
            interleaved: false
        ) else {
            throw NSError(domain: "StreamingWAVWriter", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"])
        }
        self.format = format
        self.audioFile = try AVAudioFile(forWriting: url, settings: format.settings)
    }

    /// Write a chunk of audio samples to the file
    public func writeChunk(_ samples: [Float]) throws {
        guard let file = audioFile else {
            throw WAVWriterError.fileNotOpen
        }

        let frames = samples.count
        guard frames > 0 else { return }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
            throw WAVWriterError.bufferAllocFailed
        }
        buffer.frameLength = AVAudioFrameCount(frames)

        guard let dst = buffer.floatChannelData else {
            throw WAVWriterError.bufferAllocFailed
        }

        samples.withUnsafeBufferPointer { src in
            guard let baseAddress = src.baseAddress else { return }
            dst[0].update(from: baseAddress, count: frames)
        }

        try file.write(from: buffer)
        totalFramesWritten += frames
    }

    /// Close the file and return the URL
    public func finalize() -> URL {
        audioFile = nil  // Closes the file
        return url
    }

    /// Get total frames written so far
    public var framesWritten: Int {
        return totalFramesWritten
    }

    /// Get duration in seconds
    public var duration: Double {
        return Double(totalFramesWritten) / format.sampleRate
    }
}

public func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws {
    let samples = audio.asArray(Float.self)
    try saveAudioSamples(samples, sampleRate: sampleRate, to: url)
}

/// Save audio samples directly from a Float array (avoids GPU memory copy)
public func saveAudioSamples(_ samples: [Float], sampleRate: Double, to url: URL) throws {
    let frames = samples.count
    guard frames > 0 else { throw WAVWriterError.noFrames }

    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: AVAudioChannelCount(1), interleaved: false) else {
        throw NSError(domain: "WAVWriter", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"])
    }
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
        throw WAVWriterError.bufferAllocFailed
    }
    buffer.frameLength = AVAudioFrameCount(frames)

    guard let dst = buffer.floatChannelData else {
        throw WAVWriterError.bufferAllocFailed
    }
    samples.withUnsafeBufferPointer { src in
        guard let baseAddress = src.baseAddress else { return }
        dst[0].update(from: baseAddress, count: frames)
    }

    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
}
