import SwiftUI

struct STTSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var viewModel: STTViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                settingsContent
            }
            .navigationTitle("STT Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 450, minHeight: 600)
        #endif
    }

    #if os(iOS)
    private let sectionSpacing: CGFloat = 12
    private let labelFont: Font = .caption
    private let textFont: Font = .footnote
    private let horizontalPadding: CGFloat = 16
    #else
    private let sectionSpacing: CGFloat = 16
    private let labelFont: Font = .subheadline
    private let textFont: Font = .subheadline
    private let horizontalPadding: CGFloat = 20
    #endif

    private var settingsContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Model Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Model")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack(spacing: 6) {
                    TextField("Model ID", text: $viewModel.modelId)
                        .font(textFont)
                        .textFieldStyle(.plain)
                        .padding(8)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    Button(action: {
                        Task {
                            await viewModel.reloadModel()
                        }
                    }) {
                        Text("Load")
                            .font(textFont)
                            .fontWeight(.medium)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isLoading)
                }
                .padding(.top, 4)
            }
            .padding(.bottom, sectionSpacing)

            // Max Tokens Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Length")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Max Tokens")
                        .font(textFont)
                    Spacer()
                    Text("\(viewModel.maxTokens)")
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    range: 512...16384,
                    step: 512
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    in: 512...16384,
                    step: 512
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Temperature Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Temperature")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Temperature")
                        .font(textFont)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.temperature))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.temperature) },
                        set: { viewModel.temperature = Float($0) }
                    ),
                    range: 0.0...1.0,
                    step: 0.05
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.temperature) },
                        set: { viewModel.temperature = Float($0) }
                    ),
                    in: 0.0...1.0,
                    step: 0.05
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Language Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Language")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                TextField("Language", text: $viewModel.language)
                    .font(textFont)
                    .textFieldStyle(.plain)
                    .padding(8)
                    .background(Color.gray.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .padding(.top, 4)

                Text("e.g. English, Chinese, Japanese, Korean")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.bottom, sectionSpacing)

            // Reset button
            Button(action: {
                viewModel.modelId = "mlx-community/Qwen3-ASR-0.6B-4bit"
                viewModel.maxTokens = 8192
                viewModel.temperature = 0.0
                viewModel.language = "English"
            }) {
                Text("Reset to Defaults")
                    .font(textFont)
                    .fontWeight(.medium)
                    .foregroundStyle(.blue)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.blue.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .padding(.top, 16)
            .padding(.bottom, 12)
        }
        .padding(.horizontal, horizontalPadding)
    }
}

#Preview {
    STTSettingsView(viewModel: STTViewModel())
}
