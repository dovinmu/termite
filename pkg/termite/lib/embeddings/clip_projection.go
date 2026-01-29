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

//go:build onnx && ORT

package embeddings

import (
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// projectionSession wraps an ONNX session for simple matrix projection.
// The projection model takes [batch, input_dim] and outputs [batch, output_dim].
type projectionSession struct {
	session   *ort.DynamicAdvancedSession
	inputDim  int
	outputDim int
}

// newProjectionSession loads a projection ONNX model.
// The model expects input named "input" with shape [batch, input_dim]
// and outputs "output" with shape [batch, output_dim].
func newProjectionSession(onnxPath string) (*projectionSession, error) {
	// Check if file exists
	if _, err := os.Stat(onnxPath); err != nil {
		return nil, fmt.Errorf("projection model not found: %s", onnxPath)
	}

	// Get model input/output info to determine dimensions
	inputs, outputs, err := ort.GetInputOutputInfo(onnxPath)
	if err != nil {
		return nil, fmt.Errorf("getting model info: %w", err)
	}

	if len(inputs) != 1 || len(outputs) != 1 {
		return nil, fmt.Errorf("projection model should have 1 input and 1 output, got %d inputs and %d outputs",
			len(inputs), len(outputs))
	}

	// Extract dimensions from shape (expecting [batch, dim] where batch might be dynamic)
	inputShape := inputs[0].Dimensions
	outputShape := outputs[0].Dimensions

	if len(inputShape) != 2 || len(outputShape) != 2 {
		return nil, fmt.Errorf("projection model shapes should be 2D, got input %v and output %v",
			inputShape, outputShape)
	}

	// The last dimension is the fixed dimension (batch is typically first and may be -1 for dynamic)
	inputDim := int(inputShape[1])
	outputDim := int(outputShape[1])

	// Create dynamic session that handles variable batch sizes
	session, err := ort.NewDynamicAdvancedSession(
		onnxPath,
		[]string{inputs[0].Name},
		[]string{outputs[0].Name},
		nil, // Use default session options
	)
	if err != nil {
		return nil, fmt.Errorf("creating projection session: %w", err)
	}

	return &projectionSession{
		session:   session,
		inputDim:  inputDim,
		outputDim: outputDim,
	}, nil
}

// Project applies the projection layer to the input embedding.
// Input should have length inputDim, output will have length outputDim.
func (p *projectionSession) Project(embedding []float32) ([]float32, error) {
	if len(embedding) != p.inputDim {
		return nil, fmt.Errorf("input dimension mismatch: expected %d, got %d", p.inputDim, len(embedding))
	}

	// Create input tensor with shape [1, inputDim]
	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(p.inputDim)), embedding)
	if err != nil {
		return nil, fmt.Errorf("creating input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor with shape [1, outputDim]
	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(p.outputDim)))
	if err != nil {
		return nil, fmt.Errorf("creating output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = p.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("running projection: %w", err)
	}

	// Copy output data (GetData returns slice backed by tensor memory)
	result := make([]float32, p.outputDim)
	copy(result, outputTensor.GetData())

	return result, nil
}

// InputDim returns the expected input dimension.
func (p *projectionSession) InputDim() int {
	return p.inputDim
}

// OutputDim returns the output dimension.
func (p *projectionSession) OutputDim() int {
	return p.outputDim
}

// Close releases resources.
func (p *projectionSession) Close() error {
	if p.session != nil {
		return p.session.Destroy()
	}
	return nil
}
