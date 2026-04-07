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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsSplitCausalLMModel_Gemma4Style(t *testing.T) {
	dir := t.TempDir()
	// Gemma 4 E2B Transformers.js layout
	require.NoError(t, os.WriteFile(filepath.Join(dir, "decoder_model_merged_q4f16.onnx"), []byte("decoder"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "embed_tokens_q4f16.onnx"), []byte("embed"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0644))

	assert.True(t, IsSplitCausalLMModel(dir))
}

func TestIsSplitCausalLMModel_ExactNames(t *testing.T) {
	dir := t.TempDir()
	// Exact filenames without variant suffix
	require.NoError(t, os.WriteFile(filepath.Join(dir, "decoder_model_merged.onnx"), []byte("decoder"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "embed_tokens.onnx"), []byte("embed"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0644))

	assert.True(t, IsSplitCausalLMModel(dir))
}

func TestIsSplitCausalLMModel_NotSplit_SingleModel(t *testing.T) {
	dir := t.TempDir()
	// Standard single-file generator
	require.NoError(t, os.WriteFile(filepath.Join(dir, "model.onnx"), []byte("model"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0644))

	assert.False(t, IsSplitCausalLMModel(dir))
}

func TestIsSplitCausalLMModel_NotSplit_VisionModel(t *testing.T) {
	dir := t.TempDir()
	// Vision2Seq model: has decoder + embed_tokens + vision_encoder
	require.NoError(t, os.WriteFile(filepath.Join(dir, "decoder_model_merged.onnx"), []byte("decoder"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "embed_tokens.onnx"), []byte("embed"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "vision_encoder.onnx"), []byte("vision"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0644))

	// Should be false — vision models are handled by vision2seq pipeline
	assert.False(t, IsSplitCausalLMModel(dir))
}

func TestIsSplitCausalLMModel_NotSplit_DecoderOnly(t *testing.T) {
	dir := t.TempDir()
	// Has decoder but no embed_tokens
	require.NoError(t, os.WriteFile(filepath.Join(dir, "decoder_model_merged.onnx"), []byte("decoder"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0644))

	assert.False(t, IsSplitCausalLMModel(dir))
}

func TestIsSplitCausalLMModel_NotSplit_EmptyDir(t *testing.T) {
	dir := t.TempDir()
	assert.False(t, IsSplitCausalLMModel(dir))
}
