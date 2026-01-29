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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/antflydb/antfly-go/libaf/ai"
	libafembed "github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util/imageutil"
	"go.uber.org/zap"
	_ "golang.org/x/image/webp"
)

// HugotCLIPEmbedder implements multimodal embeddings using CLIP models via hugot pipelines.
// It uses hugot's FeatureExtractionPipeline with ImageMode for the visual encoder
// and a regular text pipeline for the text encoder.
//
// Build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type HugotCLIPEmbedder struct {
	visualPipeline   *pipelines.FeatureExtractionPipeline
	textPipeline     *pipelines.FeatureExtractionPipeline
	session          *khugot.Session
	config           *HugotCLIPConfig
	logger           *zap.Logger
	caps             libafembed.EmbedderCapabilities
	modelPath        string
	sessionShared    bool
	visualProjection *projectionSession // Projects visual encoder output to shared space
	textProjection   *projectionSession // Projects text encoder output to shared space
}

// HugotCLIPConfig holds the CLIP model configuration
type HugotCLIPConfig struct {
	ModelType     string `json:"model_type"`
	ProjectionDim int    `json:"projection_dim"`
	ImageSize     int    `json:"image_size"`
}

// NewHugotCLIPEmbedder creates a new CLIP embedder using hugot pipelines.
// The directory should contain:
//   - visual_model.onnx (or visual_model_quantized.onnx)
//   - text_model.onnx (or text_model_quantized.onnx)
//   - config.json or clip_config.json
//   - tokenizer.json
//
// Build with -tags="onnx,ORT" to enable this embedder.
func NewHugotCLIPEmbedder(modelPath string, quantized bool, logger *zap.Logger) (*HugotCLIPEmbedder, error) {
	return NewHugotCLIPEmbedderWithSession(modelPath, quantized, nil, logger)
}

// NewHugotCLIPEmbedderWithSessionManager creates a new CLIP embedder using a SessionManager.
// The SessionManager handles backend selection and session reuse (required for ONNX Runtime which only allows one session).
// modelBackends restricts which backends can be used (nil = ONNX only, which is required for CLIP).
// Returns the embedder and the backend type that was used.
func NewHugotCLIPEmbedderWithSessionManager(modelPath string, quantized bool, sessionManager *hugot.SessionManager, modelBackends []string, logger *zap.Logger) (*HugotCLIPEmbedder, hugot.BackendType, error) {
	if sessionManager == nil {
		return nil, "", errors.New("sessionManager is required for CLIP embedder (ONNX Runtime only allows one session)")
	}

	// CLIP requires ONNX Runtime backend (not pure Go or XLA)
	// If modelBackends is nil, default to ONNX only
	if modelBackends == nil {
		modelBackends = []string{"onnx"}
	}
	session, backendUsed, err := sessionManager.GetSessionForModel(modelBackends)
	if err != nil {
		return nil, "", fmt.Errorf("getting session from manager: %w", err)
	}

	embedder, err := NewHugotCLIPEmbedderWithSession(modelPath, quantized, session, logger)
	if err != nil {
		return nil, "", err
	}

	// SessionManager owns the session, so mark as shared
	embedder.sessionShared = true

	return embedder, backendUsed, nil
}

// NewHugotCLIPEmbedderWithSession creates a new CLIP embedder using an optional shared session.
func NewHugotCLIPEmbedderWithSession(modelPath string, quantized bool, sharedSession *khugot.Session, logger *zap.Logger) (*HugotCLIPEmbedder, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing HugotCLIP embedder",
		zap.String("modelPath", modelPath),
		zap.Bool("quantized", quantized),
		zap.String("backend", hugot.BackendName()))

	// Load configuration
	config, err := loadHugotCLIPConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading CLIP config: %w", err)
	}

	// Determine ONNX filenames
	visualFile := "visual_model.onnx"
	textFile := "text_model.onnx"
	if quantized {
		visualFile = "visual_model_quantized.onnx"
		textFile = "text_model_quantized.onnx"
	}

	// Verify files exist
	visualPath := filepath.Join(modelPath, visualFile)
	textPath := filepath.Join(modelPath, textFile)
	if _, err := os.Stat(visualPath); err != nil {
		return nil, fmt.Errorf("visual model not found: %s", visualPath)
	}
	if _, err := os.Stat(textPath); err != nil {
		return nil, fmt.Errorf("text model not found: %s", textPath)
	}

	// Create or reuse session
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)

	// Get image size for preprocessing
	imageSize := config.ImageSize
	if imageSize == 0 {
		imageSize = 224 // Default for CLIP
	}

	// Create visual pipeline with image mode
	visualPipelineName := fmt.Sprintf("%s:visual:%s", modelPath, visualFile)
	visualConfig := khugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         visualPipelineName,
		OnnxFilename: visualFile,
		Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
			pipelines.WithImageMode(),
			pipelines.WithPreprocessSteps[*pipelines.FeatureExtractionPipeline](
				imageutil.ResizeStep(imageSize),
				imageutil.CenterCropStep(imageSize, imageSize),
			),
			pipelines.WithNormalizationSteps[*pipelines.FeatureExtractionPipeline](
				imageutil.RescaleStep(),
				imageutil.CLIPPixelNormalizationStep(),
			),
			pipelines.WithNCHWFormat[*pipelines.FeatureExtractionPipeline](),
			pipelines.WithNormalization(),
		},
	}

	visualPipeline, err := khugot.NewPipeline(session, visualConfig)
	if err != nil {
		if !sessionShared {
			_ = session.Destroy()
		}
		return nil, fmt.Errorf("creating visual pipeline: %w", err)
	}

	// Create text pipeline (regular feature extraction)
	textPipelineName := fmt.Sprintf("%s:text:%s", modelPath, textFile)
	textConfig := khugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         textPipelineName,
		OnnxFilename: textFile,
		Options: []backends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
			pipelines.WithNormalization(),
		},
	}

	textPipeline, err := khugot.NewPipeline(session, textConfig)
	if err != nil {
		if !sessionShared {
			_ = session.Destroy()
		}
		return nil, fmt.Errorf("creating text pipeline: %w", err)
	}

	// Load projection layers if they exist
	// These transform encoder outputs to the shared embedding space
	var visualProjection, textProjection *projectionSession

	visualProjPath := filepath.Join(modelPath, "visual_projection.onnx")
	textProjPath := filepath.Join(modelPath, "text_projection.onnx")

	if _, statErr := os.Stat(visualProjPath); statErr == nil {
		visualProjection, err = newProjectionSession(visualProjPath)
		if err != nil {
			logger.Warn("Failed to load visual projection layer",
				zap.String("path", visualProjPath),
				zap.Error(err))
		} else {
			logger.Info("Loaded visual projection layer",
				zap.String("path", visualProjPath),
				zap.Int("inputDim", visualProjection.InputDim()),
				zap.Int("outputDim", visualProjection.OutputDim()))
		}
	}

	if _, statErr := os.Stat(textProjPath); statErr == nil {
		textProjection, err = newProjectionSession(textProjPath)
		if err != nil {
			logger.Warn("Failed to load text projection layer",
				zap.String("path", textProjPath),
				zap.Error(err))
		} else {
			logger.Info("Loaded text projection layer",
				zap.String("path", textProjPath),
				zap.Int("inputDim", textProjection.InputDim()),
				zap.Int("outputDim", textProjection.OutputDim()))
		}
	}

	// Warn if only one projection is available (cross-modal won't work properly)
	if (visualProjection == nil) != (textProjection == nil) {
		logger.Warn("Partial projection loading - cross-modal similarity may not work correctly",
			zap.Bool("hasVisualProjection", visualProjection != nil),
			zap.Bool("hasTextProjection", textProjection != nil))
	}

	logger.Info("HugotCLIP embedder initialized",
		zap.Int("projectionDim", config.ProjectionDim),
		zap.Int("imageSize", imageSize),
		zap.Bool("hasVisualProjection", visualProjection != nil),
		zap.Bool("hasTextProjection", textProjection != nil))

	return &HugotCLIPEmbedder{
		visualPipeline:   visualPipeline,
		textPipeline:     textPipeline,
		session:          session,
		config:           config,
		logger:           logger,
		modelPath:        modelPath,
		sessionShared:    sessionShared,
		visualProjection: visualProjection,
		textProjection:   textProjection,
		caps: libafembed.EmbedderCapabilities{
			SupportedMIMETypes: []libafembed.MIMETypeSupport{
				{MIMEType: "text/plain"},
				{MIMEType: "image/png"},
				{MIMEType: "image/jpeg"},
				{MIMEType: "image/gif"},
				{MIMEType: "image/webp"},
			},
			Dimensions:       []int{config.ProjectionDim},
			DefaultDimension: config.ProjectionDim,
			SupportsFusion:   false,
		},
	}, nil
}

// Capabilities returns the embedder capabilities
func (c *HugotCLIPEmbedder) Capabilities() libafembed.EmbedderCapabilities {
	return c.caps
}

// Embed generates embeddings for the given content.
// For text content, uses the text encoder.
// For image content (BinaryContent), uses the visual encoder.
func (c *HugotCLIPEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	if len(contents) == 0 {
		return [][]float32{}, nil
	}

	embeddings := make([][]float32, len(contents))

	for i, parts := range contents {
		var embedding []float32
		var err error

		for _, part := range parts {
			switch p := part.(type) {
			case ai.BinaryContent:
				if strings.HasPrefix(p.MIMEType, "image/") {
					embedding, err = c.embedImage(p.Data)
					if err != nil {
						return nil, fmt.Errorf("embedding image at index %d: %w", i, err)
					}
				}
			case ai.TextContent:
				embedding, err = c.embedText(p.Text)
				if err != nil {
					return nil, fmt.Errorf("embedding text at index %d: %w", i, err)
				}
			}

			if embedding != nil {
				break
			}
		}

		if embedding == nil {
			return nil, fmt.Errorf("no valid content found at index %d", i)
		}

		embeddings[i] = embedding
	}

	return embeddings, nil
}

// embedImage processes an image and returns its embedding using the visual pipeline
func (c *HugotCLIPEmbedder) embedImage(imageData []byte) ([]float32, error) {
	// Decode image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("decoding image: %w", err)
	}

	// Run through visual pipeline
	output, err := c.visualPipeline.RunWithImages([]image.Image{img})
	if err != nil {
		return nil, fmt.Errorf("running visual pipeline: %w", err)
	}

	if len(output.Embeddings) == 0 || len(output.Embeddings[0]) == 0 {
		return nil, errors.New("no embedding returned from visual pipeline")
	}

	embedding := output.Embeddings[0]

	// Apply visual projection if available
	// This transforms the encoder output (e.g., 768-dim) to the shared embedding space (e.g., 512-dim)
	if c.visualProjection != nil {
		projected, err := c.visualProjection.Project(embedding)
		if err != nil {
			return nil, fmt.Errorf("applying visual projection: %w", err)
		}
		embedding = projected
	}

	// Normalize after projection
	return normalizeEmbedding(embedding), nil
}

// embedText tokenizes text and returns its embedding using the text pipeline
func (c *HugotCLIPEmbedder) embedText(text string) ([]float32, error) {
	// Run through text pipeline
	output, err := c.textPipeline.RunPipeline([]string{text})
	if err != nil {
		return nil, fmt.Errorf("running text pipeline: %w", err)
	}

	if len(output.Embeddings) == 0 || len(output.Embeddings[0]) == 0 {
		return nil, errors.New("no embedding returned from text pipeline")
	}

	embedding := output.Embeddings[0]

	// Apply text projection if available
	// This transforms the encoder output to the shared embedding space
	if c.textProjection != nil {
		projected, err := c.textProjection.Project(embedding)
		if err != nil {
			return nil, fmt.Errorf("applying text projection: %w", err)
		}
		embedding = projected
	}

	// Normalize after projection
	return normalizeEmbedding(embedding), nil
}

// Close releases resources
func (c *HugotCLIPEmbedder) Close() error {
	var errs []error

	// Close projection sessions
	if c.visualProjection != nil {
		if err := c.visualProjection.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing visual projection: %w", err))
		}
	}
	if c.textProjection != nil {
		if err := c.textProjection.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing text projection: %w", err))
		}
	}

	// Close hugot session if owned
	if c.session != nil && !c.sessionShared {
		c.logger.Info("Destroying Hugot session (owned by this CLIP embedder)")
		if err := c.session.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("destroying session: %w", err))
		}
	} else if c.sessionShared {
		c.logger.Debug("Skipping session destruction (shared session)")
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing CLIP embedder: %v", errs)
	}
	return nil
}

// normalizeEmbedding applies L2 normalization to an embedding vector.
// This ensures the embedding has unit length, which is required for cosine similarity.
func normalizeEmbedding(v []float32) []float32 {
	var sumSquares float64
	for _, x := range v {
		sumSquares += float64(x) * float64(x)
	}
	norm := math.Sqrt(sumSquares)
	if norm == 0 {
		return v
	}
	result := make([]float32, len(v))
	for i, x := range v {
		result[i] = float32(float64(x) / norm)
	}
	return result
}

// loadHugotCLIPConfig loads CLIP configuration from model directory
func loadHugotCLIPConfig(modelPath string) (*HugotCLIPConfig, error) {
	configPaths := []string{
		filepath.Join(modelPath, "clip_config.json"),
		filepath.Join(modelPath, "config.json"),
	}

	for _, path := range configPaths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		// Try to parse as CLIP config with nested structure
		var nestedConfig struct {
			ModelType     string `json:"model_type"`
			ProjectionDim int    `json:"projection_dim"`
			VisionConfig  struct {
				ImageSize     int `json:"image_size"`
				ProjectionDim int `json:"projection_dim"`
			} `json:"vision_config"`
		}

		if err := json.Unmarshal(data, &nestedConfig); err == nil {
			projDim := nestedConfig.ProjectionDim
			if projDim == 0 {
				projDim = nestedConfig.VisionConfig.ProjectionDim
			}
			if projDim == 0 {
				projDim = 512 // Default CLIP dimension
			}

			imageSize := nestedConfig.VisionConfig.ImageSize
			if imageSize == 0 {
				imageSize = 224 // Default
			}

			return &HugotCLIPConfig{
				ModelType:     nestedConfig.ModelType,
				ProjectionDim: projDim,
				ImageSize:     imageSize,
			}, nil
		}
	}

	// Return default config for CLIP ViT-B/32
	return &HugotCLIPConfig{
		ModelType:     "clip",
		ProjectionDim: 512,
		ImageSize:     224,
	}, nil
}
