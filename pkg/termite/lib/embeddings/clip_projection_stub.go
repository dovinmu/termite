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

//go:build !onnx || !ORT

package embeddings

import "errors"

// projectionSession is a stub for non-ONNX builds.
type projectionSession struct{}

func newProjectionSession(onnxPath string) (*projectionSession, error) {
	return nil, errors.New("projection sessions require ONNX Runtime (build with -tags='onnx,ORT')")
}

func (p *projectionSession) Project(embedding []float32) ([]float32, error) {
	return nil, errors.New("projection sessions require ONNX Runtime")
}

func (p *projectionSession) InputDim() int  { return 0 }
func (p *projectionSession) OutputDim() int { return 0 }
func (p *projectionSession) Close() error   { return nil }
