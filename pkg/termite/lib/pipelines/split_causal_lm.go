// Copyright 2025 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pipelines

import (
	"context"
	"fmt"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
)

// splitCausalLMDecoderCandidates are filenames to search for the decoder ONNX file.
var splitCausalLMDecoderCandidates = []string{
	"decoder_model_merged.onnx",
	"decoder_model.onnx",
}

// splitCausalLMEmbedTokensCandidates are filenames to search for the embed_tokens ONNX file.
var splitCausalLMEmbedTokensCandidates = []string{
	"embed_tokens.onnx",
}

// splitCausalLMVisionEncoderCandidates are filenames that indicate a vision model
// (not a text-only split causal LM).
var splitCausalLMVisionEncoderCandidates = []string{
	"vision_encoder.onnx",
	"audio_encoder.onnx",
}

// IsSplitCausalLMModel checks if a model path contains a split causal LM
// (embed_tokens + decoder, without vision/audio encoders).
// This detects Transformers.js-style ONNX exports like onnx-community/gemma-4-E2B-it-ONNX.
func IsSplitCausalLMModel(path string) bool {
	hasDecoder := FindONNXFile(path, splitCausalLMDecoderCandidates) != ""
	hasEmbedTokens := FindONNXFile(path, splitCausalLMEmbedTokensCandidates) != ""
	if !hasDecoder || !hasEmbedTokens {
		return false
	}
	// Exclude vision/audio models — those are handled by the vision2seq pipeline
	hasVisionOrAudio := FindONNXFile(path, splitCausalLMVisionEncoderCandidates) != ""
	return !hasVisionOrAudio
}

// Ensure splitCausalLMModel implements backends.Model
var _ backends.Model = (*splitCausalLMModel)(nil)

// splitCausalLMModel implements backends.Model for split causal LM models
// where token embedding and decoding are separate ONNX sessions.
// This supports Transformers.js-style ONNX exports (e.g., Gemma 4 E2B)
// with the inference flow: input_ids → embed_tokens → decoder → logits.
type splitCausalLMModel struct {
	config *GenerativeModelConfig

	embedTokensSession backends.Session // embed_tokens_*.onnx
	decoderSession     backends.Session // decoder_model_merged_*.onnx

	backendType backends.BackendType
}

// LoadSplitCausalLMModel loads a split causal LM model from a directory.
func LoadSplitCausalLMModel(modelPath string, factory backends.SessionFactory) (backends.Model, error) {
	config, err := LoadGenerativeModelConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading model config: %w", err)
	}

	embedTokensPath := FindONNXFile(modelPath, splitCausalLMEmbedTokensCandidates)
	if embedTokensPath == "" {
		return nil, fmt.Errorf("embed_tokens ONNX file not found in %s", modelPath)
	}

	decoderPath := FindONNXFile(modelPath, splitCausalLMDecoderCandidates)
	if decoderPath == "" {
		return nil, fmt.Errorf("decoder ONNX file not found in %s", modelPath)
	}

	embedTokensSession, err := factory.CreateSession(embedTokensPath)
	if err != nil {
		return nil, fmt.Errorf("creating embed_tokens session: %w", err)
	}

	decoderSession, err := factory.CreateSession(decoderPath)
	if err != nil {
		_ = embedTokensSession.Close()
		return nil, fmt.Errorf("creating decoder session: %w", err)
	}

	return &splitCausalLMModel{
		config:             config,
		embedTokensSession: embedTokensSession,
		decoderSession:     decoderSession,
		backendType:        factory.Backend(),
	}, nil
}

// Forward performs one decoding step:
// 1. Run embed_tokens on input_ids → inputs_embeds (+ optional per_layer_inputs)
// 2. Run decoder on inputs_embeds → logits + KV-cache
func (m *splitCausalLMModel) Forward(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error) {
	if inputs == nil || len(inputs.InputIDs) == 0 {
		return nil, fmt.Errorf("empty input")
	}

	inputIDs := inputs.InputIDs
	pastKV := inputs.PastKeyValues
	batchSize := len(inputIDs)
	seqLen := len(inputIDs[0])

	// Step 1: Run embed_tokens to get inputs_embeds
	flatInputIDs := make([]int64, batchSize*seqLen)
	for i := range batchSize {
		for j := range seqLen {
			flatInputIDs[i*seqLen+j] = int64(inputIDs[i][j])
		}
	}

	embedInputs := []backends.NamedTensor{{
		Name:  "input_ids",
		Shape: []int64{int64(batchSize), int64(seqLen)},
		Data:  flatInputIDs,
	}}

	embedOutputs, err := m.embedTokensSession.Run(embedInputs)
	if err != nil {
		return nil, fmt.Errorf("running embed_tokens: %w", err)
	}
	if len(embedOutputs) == 0 {
		return nil, fmt.Errorf("no embed_tokens output")
	}

	// Step 2: Build decoder inputs
	decoderInputs, err := m.buildDecoderInputs(embedOutputs, batchSize, seqLen, pastKV)
	if err != nil {
		return nil, fmt.Errorf("building decoder inputs: %w", err)
	}

	// Step 3: Run decoder
	decoderOutputs, err := m.decoderSession.Run(decoderInputs)
	if err != nil {
		return nil, fmt.Errorf("running decoder: %w", err)
	}
	if len(decoderOutputs) == 0 {
		return nil, fmt.Errorf("no decoder output")
	}

	// Step 4: Extract logits and KV-cache
	logitsOutput := decoderOutputs[0]
	logitsData, ok := logitsOutput.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("logits tensor is not float32")
	}

	logitsShape := logitsOutput.Shape
	vocabSize := int(logitsShape[len(logitsShape)-1])
	outputSeqLen := 1
	if len(logitsShape) == 3 {
		outputSeqLen = int(logitsShape[1])
	}

	logits := make([][]float32, batchSize)
	for i := range batchSize {
		logits[i] = make([]float32, vocabSize)
		// Take logits from last position
		startIdx := i*outputSeqLen*vocabSize + (outputSeqLen-1)*vocabSize
		copy(logits[i], logitsData[startIdx:startIdx+vocabSize])
	}

	newKVCache := m.extractKVCache(decoderOutputs, batchSize, pastKV)

	return &backends.ModelOutput{
		Logits:        logits,
		PastKeyValues: newKVCache,
	}, nil
}

// buildDecoderInputs constructs the input tensors for the decoder session.
func (m *splitCausalLMModel) buildDecoderInputs(
	embedOutputs []backends.NamedTensor,
	batchSize, seqLen int,
	pastKV *backends.KVCache,
) ([]backends.NamedTensor, error) {
	var inputs []backends.NamedTensor

	// Get decoder's expected input names
	inputInfo := m.decoderSession.InputInfo()
	inputNames := make(map[string]bool)
	for _, info := range inputInfo {
		inputNames[info.Name] = true
	}

	// Add embed_tokens outputs as decoder inputs.
	// The first output is always inputs_embeds.
	// Some models (Gemma 4 with PLE) also output per_layer_inputs.
	for _, output := range embedOutputs {
		if inputNames[output.Name] {
			inputs = append(inputs, backends.NamedTensor{
				Name:  output.Name,
				Shape: output.Shape,
				Data:  output.Data,
			})
		}
	}

	// Add attention_mask
	if inputNames["attention_mask"] {
		totalSeqLen := seqLen
		if pastKV != nil {
			totalSeqLen = pastKV.SeqLen + seqLen
		}
		mask := make([]int64, batchSize*totalSeqLen)
		for i := range mask {
			mask[i] = 1
		}
		inputs = append(inputs, backends.NamedTensor{
			Name:  "attention_mask",
			Shape: []int64{int64(batchSize), int64(totalSeqLen)},
			Data:  mask,
		})
	}

	// Add position_ids
	if inputNames["position_ids"] {
		startPos := 0
		if pastKV != nil {
			startPos = pastKV.SeqLen
		}
		posIDs := make([]int64, batchSize*seqLen)
		for i := range batchSize {
			for j := range seqLen {
				posIDs[i*seqLen+j] = int64(startPos + j)
			}
		}
		inputs = append(inputs, backends.NamedTensor{
			Name:  "position_ids",
			Shape: []int64{int64(batchSize), int64(seqLen)},
			Data:  posIDs,
		})
	}

	// Add num_logits_to_keep (optimization: only compute logits for last token)
	if inputNames["num_logits_to_keep"] {
		inputs = append(inputs, backends.NamedTensor{
			Name:  "num_logits_to_keep",
			Shape: []int64{1},
			Data:  []int64{1},
		})
	}

	// Add use_cache_branch if needed
	if inputNames["use_cache_branch"] {
		useCache := []float32{0}
		if pastKV != nil && pastKV.SeqLen > 0 {
			useCache[0] = 1
		}
		inputs = append(inputs, backends.NamedTensor{
			Name:  "use_cache_branch",
			Shape: []int64{1},
			Data:  useCache,
		})
	}

	// Add past_key_values inputs
	for _, info := range inputInfo {
		if IsPastKeyValueInput(info.Name) {
			tensor := m.createPastKVTensor(info.Name, pastKV, batchSize)
			inputs = append(inputs, tensor)
		}
	}

	return inputs, nil
}

// createPastKVTensor creates a tensor for past key/value cache.
func (m *splitCausalLMModel) createPastKVTensor(name string, pastKV *backends.KVCache, batchSize int) backends.NamedTensor {
	if pastKV != nil && pastKV.SeqLen > 0 && pastKV.Tensors != nil {
		outputName := mapPastToPresent(name)
		if tensor, ok := pastKV.Tensors[outputName]; ok {
			return backends.NamedTensor{
				Name:  name,
				Shape: tensor.Shape,
				Data:  tensor.Data,
			}
		}
	}

	// First step — create zero-sized tensor [batch, num_heads, 0, head_dim]
	numHeads := m.config.NumKVHeads
	if numHeads == 0 {
		numHeads = m.config.NumHeads
	}
	return backends.NamedTensor{
		Name:  name,
		Shape: []int64{int64(batchSize), int64(numHeads), 0, int64(m.config.HeadDim)},
		Data:  []float32{},
	}
}

// extractKVCache extracts the KV-cache from decoder outputs.
func (m *splitCausalLMModel) extractKVCache(outputs []backends.NamedTensor, batchSize int, pastKV *backends.KVCache) *backends.KVCache {
	tensors := make(map[string]backends.NamedTensor)
	hasKVOutputs := false

	for _, output := range outputs {
		if IsPresentKeyValueOutput(output.Name) {
			hasKVOutputs = true
			data, ok := output.Data.([]float32)
			if ok {
				dataCopy := make([]float32, len(data))
				copy(dataCopy, data)
				shapeCopy := make([]int64, len(output.Shape))
				copy(shapeCopy, output.Shape)
				tensors[output.Name] = backends.NamedTensor{
					Name:  output.Name,
					Shape: shapeCopy,
					Data:  dataCopy,
				}
			}
		}
	}

	numKVHeads := m.config.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = m.config.NumHeads
	}

	if hasKVOutputs {
		seqLen := 0
		for _, tensor := range tensors {
			if len(tensor.Shape) >= 3 {
				seqLen = int(tensor.Shape[2])
				break
			}
		}
		return &backends.KVCache{
			SeqLen:    seqLen,
			NumLayers: m.config.NumLayers,
			NumHeads:  numKVHeads,
			HeadDim:   m.config.HeadDim,
			BatchSize: batchSize,
			Tensors:   tensors,
		}
	}

	if pastKV != nil {
		return &backends.KVCache{
			SeqLen:    pastKV.SeqLen + 1,
			NumLayers: m.config.NumLayers,
			NumHeads:  numKVHeads,
			HeadDim:   m.config.HeadDim,
			BatchSize: batchSize,
			Tensors:   pastKV.Tensors,
		}
	}

	return &backends.KVCache{
		SeqLen:    1,
		NumLayers: m.config.NumLayers,
		NumHeads:  numKVHeads,
		HeadDim:   m.config.HeadDim,
		BatchSize: batchSize,
	}
}

// DecoderConfig returns configuration needed for generation.
func (m *splitCausalLMModel) DecoderConfig() *backends.DecoderConfig {
	return m.config.DecoderConfig
}

// Close releases resources.
func (m *splitCausalLMModel) Close() error {
	var firstErr error
	if m.embedTokensSession != nil {
		if err := m.embedTokensSession.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("closing embed_tokens: %w", err)
		}
		m.embedTokensSession = nil
	}
	if m.decoderSession != nil {
		if err := m.decoderSession.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("closing decoder: %w", err)
		}
		m.decoderSession = nil
	}
	return firstErr
}

// Name returns the model name.
func (m *splitCausalLMModel) Name() string {
	return m.config.ModelPath
}

// Backend returns the backend type.
func (m *splitCausalLMModel) Backend() backends.BackendType {
	return m.backendType
}
