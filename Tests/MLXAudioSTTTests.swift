//
//  MLXAudioSTTTests.swift
//  MLXAudioTests
//
//  Created by Prince Canuma on 04/01/2026.
//

import Testing
import XCTest
import MLX
import MLXNN

@testable import MLXAudioCore
@testable import MLXAudioSTT


final class GLMASRModuleTests: XCTestCase {

    // MARK: - Configuration Tests

    func testWhisperConfigDefaults() {
        let config = WhisperConfig()

        XCTAssertEqual(config.modelType, "whisper")
        XCTAssertEqual(config.activationFunction, "gelu")
        XCTAssertEqual(config.dModel, 1280)
        XCTAssertEqual(config.encoderAttentionHeads, 20)
        XCTAssertEqual(config.encoderFfnDim, 5120)
        XCTAssertEqual(config.encoderLayers, 32)
        XCTAssertEqual(config.numMelBins, 128)
        XCTAssertEqual(config.maxSourcePositions, 1500)
        XCTAssertTrue(config.ropeTraditional)
    }

    func testWhisperConfigCustom() {
        let config = WhisperConfig(
            dModel: 512,
            encoderAttentionHeads: 8,
            encoderLayers: 6,
            numMelBins: 80
        )

        XCTAssertEqual(config.dModel, 512)
        XCTAssertEqual(config.encoderAttentionHeads, 8)
        XCTAssertEqual(config.encoderLayers, 6)
        XCTAssertEqual(config.numMelBins, 80)
    }

    func testLlamaConfigDefaults() {
        let config = LlamaConfig()

        XCTAssertEqual(config.modelType, "llama")
        XCTAssertEqual(config.vocabSize, 59264)
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.intermediateSize, 6144)
        XCTAssertEqual(config.numHiddenLayers, 28)
        XCTAssertEqual(config.numAttentionHeads, 16)
        XCTAssertEqual(config.numKeyValueHeads, 4)
        XCTAssertEqual(config.hiddenAct, "silu")
        XCTAssertEqual(config.eosTokenId, [59246, 59253, 59255])
    }

    func testLlamaConfigCustom() {
        let config = LlamaConfig(
            vocabSize: 32000,
            hiddenSize: 1024,
            numHiddenLayers: 12
        )

        XCTAssertEqual(config.vocabSize, 32000)
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numHiddenLayers, 12)
    }

    func testGLMASRModelConfigDefaults() {
        let config = GLMASRModelConfig()

        XCTAssertEqual(config.modelType, "glmasr")
        XCTAssertEqual(config.adapterType, "mlp")
        XCTAssertEqual(config.mergeFactor, 4)
        XCTAssertTrue(config.useRope)
        XCTAssertEqual(config.maxWhisperLength, 1500)
    }

    func testGLMASRModelConfigWithNestedConfigs() {
        let whisperConfig = WhisperConfig(dModel: 512, encoderLayers: 6)
        let llamaConfig = LlamaConfig(hiddenSize: 1024, numHiddenLayers: 12)

        let config = GLMASRModelConfig(
            whisperConfig: whisperConfig,
            lmConfig: llamaConfig,
            mergeFactor: 2
        )

        XCTAssertEqual(config.whisperConfig.dModel, 512)
        XCTAssertEqual(config.whisperConfig.encoderLayers, 6)
        XCTAssertEqual(config.lmConfig.hiddenSize, 1024)
        XCTAssertEqual(config.lmConfig.numHiddenLayers, 12)
        XCTAssertEqual(config.mergeFactor, 2)
    }

    // MARK: - Layer Tests

    func testWhisperAttentionShape() {
        let config = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderLayers: 2
        )

        let attention = WhisperAttention(config: config, useRope: false)

        let batchSize = 2
        let seqLen = 10
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.dModel])

        let output = attention(hiddenStates)

        XCTAssertEqual(output.shape, [batchSize, seqLen, config.dModel])
    }

    func testWhisperAttentionWithRoPE() {
        let config = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderLayers: 2,
            ropeTraditional: true
        )

        let attention = WhisperAttention(config: config, useRope: true)

        let batchSize = 2
        let seqLen = 10
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.dModel])

        let output = attention(hiddenStates)

        XCTAssertEqual(output.shape, [batchSize, seqLen, config.dModel])
    }

    func testWhisperEncoderLayerShape() {
        let config = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderFfnDim: 1024,
            encoderLayers: 1
        )

        let layer = WhisperEncoderLayer(config: config, useRope: false)

        let batchSize = 2
        let seqLen = 10
        let hiddenStates = MLXArray.ones([batchSize, seqLen, config.dModel])

        let output = layer(hiddenStates)

        XCTAssertEqual(output.shape, [batchSize, seqLen, config.dModel])
    }

    func testWhisperEncoderShape() {
        let config = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderFfnDim: 1024,
            encoderLayers: 2,
            numMelBins: 80,
            maxSourcePositions: 100
        )

        let encoder = WhisperEncoder(config: config, useRope: false)

        let batchSize = 2
        let seqLen = 100
        let inputFeatures = MLXArray.ones([batchSize, seqLen, config.numMelBins])

        let output = encoder(inputFeatures)

        // After conv2 with stride 2, sequence length is halved
        let expectedSeqLen = seqLen / 2
        XCTAssertEqual(output.shape[0], batchSize)
        XCTAssertEqual(output.shape[1], expectedSeqLen)
        XCTAssertEqual(output.shape[2], config.dModel)
    }

    func testAdaptingMLPShape() {
        let inputDim = 512
        let intermediateDim = 1024
        let outputDim = 256

        let mlp = AdaptingMLP(inputDim: inputDim, intermediateDim: intermediateDim, outputDim: outputDim)

        let batchSize = 2
        let seqLen = 10
        let input = MLXArray.ones([batchSize, seqLen, inputDim])

        let output = mlp(input)

        XCTAssertEqual(output.shape, [batchSize, seqLen, outputDim])
    }

    func testAudioEncoderShape() {
        let whisperConfig = WhisperConfig(
            dModel: 256,
            encoderAttentionHeads: 4,
            encoderFfnDim: 1024,
            encoderLayers: 2,
            numMelBins: 80,
            maxSourcePositions: 100
        )

        let llamaConfig = LlamaConfig(
            hiddenSize: 512,
            numHiddenLayers: 2
        )

        let config = GLMASRModelConfig(
            whisperConfig: whisperConfig,
            lmConfig: llamaConfig,
            mergeFactor: 4,
            maxWhisperLength: 100
        )

        let audioEncoder = AudioEncoder(config: config)

        let batchSize = 2
        let seqLen = 100
        let inputFeatures = MLXArray.ones([batchSize, seqLen, whisperConfig.numMelBins])

        let (output, audioLen) = audioEncoder(inputFeatures)

        XCTAssertEqual(output.shape[0], batchSize)
        XCTAssertEqual(output.shape[2], llamaConfig.hiddenSize)
        XCTAssertGreaterThan(audioLen, 0)
    }

    func testAudioEncoderBoaEoaTokens() {
        let whisperConfig = WhisperConfig(dModel: 256, encoderAttentionHeads: 4, encoderLayers: 1)
        let llamaConfig = LlamaConfig(hiddenSize: 512)
        let config = GLMASRModelConfig(whisperConfig: whisperConfig, lmConfig: llamaConfig)

        let audioEncoder = AudioEncoder(config: config)

        let (boa, eoa) = audioEncoder.getBoaEoaTokens()

        XCTAssertEqual(boa.shape, [1, llamaConfig.hiddenSize])
        XCTAssertEqual(eoa.shape, [1, llamaConfig.hiddenSize])
    }

    // MARK: - STTOutput Tests

    func testSTTOutputCreation() {
        let output = STTOutput(
            text: "Hello world",
            promptTokens: 100,
            generationTokens: 50,
            totalTokens: 150,
            promptTps: 100.0,
            generationTps: 50.0,
            totalTime: 1.5
        )

        XCTAssertEqual(output.text, "Hello world")
        XCTAssertEqual(output.promptTokens, 100)
        XCTAssertEqual(output.generationTokens, 50)
        XCTAssertEqual(output.totalTokens, 150)
        XCTAssertEqual(output.promptTps, 100.0)
        XCTAssertEqual(output.generationTps, 50.0)
        XCTAssertEqual(output.totalTime, 1.5)
    }

    func testSTTOutputDefaults() {
        let output = STTOutput(text: "Test")

        XCTAssertEqual(output.text, "Test")
        XCTAssertNil(output.segments)
        XCTAssertNil(output.language)
        XCTAssertEqual(output.promptTokens, 0)
        XCTAssertEqual(output.generationTokens, 0)
        XCTAssertEqual(output.totalTokens, 0)
    }

    func testSTTOutputDescription() {
        let output = STTOutput(
            text: "Test transcription",
            language: "en",
            promptTokens: 50,
            generationTokens: 25,
            totalTokens: 75,
            totalTime: 0.5
        )

        let description = output.description

        XCTAssertTrue(description.contains("Test transcription"))
        XCTAssertTrue(description.contains("en"))
        XCTAssertTrue(description.contains("50"))
        XCTAssertTrue(description.contains("25"))
        XCTAssertTrue(description.contains("75"))
    }

    // MARK: - Config Decoding Tests

    func testWhisperConfigDecoding() throws {
        let json = """
        {
            "model_type": "whisper",
            "d_model": 512,
            "encoder_attention_heads": 8,
            "encoder_layers": 6,
            "num_mel_bins": 80
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(WhisperConfig.self, from: data)

        XCTAssertEqual(config.modelType, "whisper")
        XCTAssertEqual(config.dModel, 512)
        XCTAssertEqual(config.encoderAttentionHeads, 8)
        XCTAssertEqual(config.encoderLayers, 6)
        XCTAssertEqual(config.numMelBins, 80)
    }

    func testLlamaConfigDecoding() throws {
        let json = """
        {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "eos_token_id": [1, 2, 3]
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(LlamaConfig.self, from: data)

        XCTAssertEqual(config.modelType, "llama")
        XCTAssertEqual(config.vocabSize, 32000)
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numHiddenLayers, 12)
        XCTAssertEqual(config.eosTokenId, [1, 2, 3])
    }

    func testGLMASRModelConfigDecoding() throws {
        let json = """
        {
            "model_type": "glmasr",
            "adapter_type": "mlp",
            "merge_factor": 2,
            "use_rope": true,
            "whisper_config": {
                "d_model": 512,
                "encoder_layers": 6
            },
            "lm_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12
            }
        }
        """

        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(GLMASRModelConfig.self, from: data)

        XCTAssertEqual(config.modelType, "glmasr")
        XCTAssertEqual(config.adapterType, "mlp")
        XCTAssertEqual(config.mergeFactor, 2)
        XCTAssertTrue(config.useRope)
        XCTAssertEqual(config.whisperConfig.dModel, 512)
        XCTAssertEqual(config.whisperConfig.encoderLayers, 6)
        XCTAssertEqual(config.lmConfig.hiddenSize, 1024)
        XCTAssertEqual(config.lmConfig.numHiddenLayers, 12)
    }

    // MARK: - AnyCodable Tests

    func testAnyCodableWithInt() throws {
        let json = """
        {"value": 42}
        """

        struct Container: Codable {
            let value: AnyCodable
        }

        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        XCTAssertEqual(container.value.value as? Int, 42)
    }

    func testAnyCodableWithString() throws {
        let json = """
        {"value": "hello"}
        """

        struct Container: Codable {
            let value: AnyCodable
        }

        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        XCTAssertEqual(container.value.value as? String, "hello")
    }

    func testAnyCodableWithArray() throws {
        let json = """
        {"value": [1, 2, 3]}
        """

        struct Container: Codable {
            let value: AnyCodable
        }

        let data = json.data(using: .utf8)!
        let container = try JSONDecoder().decode(Container.self, from: data)

        let array = container.value.value as? [Any]
        XCTAssertEqual(array?.count, 3)
    }
}


// Run GLMASR tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/GLMASRTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run|Transcription)"


struct GLMASRTests {

    /// Test basic transcription with GLM-ASR model
    @Test func testGLMASRTranscribe() async throws {
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

        print("\u{001B}[33mLoading GLMASR model...\u{001B}[0m")
        let model = try await GLMASRModel.fromPretrained("mlx-community/GLM-ASR-Nano-2512-4bit")
        print("\u{001B}[32mGLMASR model loaded!\u{001B}[0m")

        let output = model.generate(audio: audioData)
        print("\u{001B}[32m GLMASR Transcription: \(output.text)\u{001B}[0m")
        print("\u{001B}[32m GLMASR Generation Stats: \(output)\u{001B}[0m")

        #expect(output.generationTokens > 0, "Generation tokens should be greater than 0")
    }

    /// Test streaming transcription with GLM-ASR model
    @Test func testGLMASRTranscribeStream() async throws {
        let audioURL = Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media")!
        let (sampleRate, audioData) = try loadAudioArray(from: audioURL)
        print("\u{001B}[33mLoaded audio: \(audioData.shape), sample rate: \(sampleRate)\u{001B}[0m")

        print("\u{001B}[33mLoading GLMASR model...\u{001B}[0m")
        let model = try await GLMASRModel.fromPretrained("mlx-community/GLM-ASR-Nano-2512-4bit")
        print("\u{001B}[32mGLMASR model loaded!\u{001B}[0m")

        print("\u{001B}[33mStreaming transcription ...\u{001B}[0m")

        var tokenCount = 0
        var transcribedText = ""
        var finalOutput: STTOutput?
        var generationInfo: STTGenerationInfo?

        for try await event in model.generateStream(audio: audioData) {
            switch event {
            case .token(let token):
                tokenCount += 1
                transcribedText += token
            case .info(let info):
                generationInfo = info
                print("\n\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .result(let output):
                finalOutput = output
                print("\u{001B}[32m GLMASR Streaming Transcription: \(output.text)\u{001B}[0m")
                print("\u{001B}[32m GLMASR Streaming Stats: \(output)\u{001B}[0m")
            }
        }

        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalOutput != nil, "Should have received final output")
        #expect(generationInfo != nil, "Should have received generation info")

        if let output = finalOutput {
            #expect(output.generationTokens > 0, "Generation tokens should be greater than 0")
            print("\u{001B}[32m\(output)\u{001B}[0m")
        }
    }
}