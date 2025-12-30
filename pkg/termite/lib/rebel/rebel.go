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

// Package rebel provides REBEL relation extraction model support.
// REBEL (Relation Extraction By End-to-end Language generation) uses a seq2seq
// architecture to extract relation triplets (subject, relation, object) from text.
//
// REBEL implements ner.Recognizer so it can be used alongside GLiNER for relation
// extraction via the /api/recognize endpoint with the "relations" capability.
package rebel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"github.com/antflydb/termite/pkg/termite/lib/ner"
	khugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	"go.uber.org/zap"
)

// Ensure HugotREBEL implements ner.Recognizer
var _ ner.Recognizer = (*HugotREBEL)(nil)

// Config holds configuration for REBEL models.
type Config struct {
	// ModelID is the original HuggingFace model ID.
	ModelID string `json:"model_id"`
	// ModelType should be "rebel".
	ModelType string `json:"model_type"`
	// MaxLength is the maximum number of tokens to generate.
	MaxLength int `json:"max_length"`
	// NumBeams is the number of beams for beam search.
	NumBeams int `json:"num_beams"`
	// Task is the model task (e.g., "relation_extraction").
	Task string `json:"task"`
	// TripletToken is the token marking triplet boundaries (default: "<triplet>").
	TripletToken string `json:"triplet_token"`
	// SubjectToken is the token marking subject boundaries (default: "<subj>").
	SubjectToken string `json:"subject_token"`
	// ObjectToken is the token marking object boundaries (default: "<obj>").
	ObjectToken string `json:"object_token"`
	// Multilingual indicates if this is a multilingual model.
	Multilingual bool `json:"multilingual"`
}

// DefaultConfig returns the default REBEL configuration.
func DefaultConfig() Config {
	return Config{
		MaxLength:    256,
		NumBeams:     3,
		TripletToken: "<triplet>",
		SubjectToken: "<subj>",
		ObjectToken:  "<obj>",
		Task:         "relation_extraction",
	}
}

// HugotREBEL implements relation extraction using REBEL via Hugot ONNX runtime.
type HugotREBEL struct {
	session       *khugot.Session
	pipeline      *pipelines.Seq2SeqPipeline
	logger        *zap.Logger
	sessionShared bool
	config        Config
}

// NewHugotREBEL creates a new REBEL model using the Hugot ONNX runtime.
func NewHugotREBEL(modelPath string, logger *zap.Logger) (*HugotREBEL, error) {
	return NewHugotREBELWithSession(modelPath, nil, logger)
}

// NewHugotREBELWithSession creates a new REBEL model with an optional shared session.
func NewHugotREBELWithSession(modelPath string, sharedSession *khugot.Session, logger *zap.Logger) (*HugotREBEL, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing Hugot REBEL model",
		zap.String("modelPath", modelPath),
		zap.String("backend", hugot.BackendName()))

	// Load REBEL config
	config := DefaultConfig()
	configPath := filepath.Join(modelPath, "rebel_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse REBEL config", zap.Error(err))
		} else {
			logger.Info("Loaded REBEL config",
				zap.String("model_id", config.ModelID),
				zap.Int("max_length", config.MaxLength),
				zap.Bool("multilingual", config.Multilingual))
		}
	}

	// Use shared session or create a new one
	session, err := hugot.NewSessionOrUseExisting(sharedSession)
	if err != nil {
		logger.Error("Failed to create Hugot session", zap.Error(err))
		return nil, fmt.Errorf("creating hugot session: %w", err)
	}
	sessionShared := (sharedSession != nil)

	// Create Seq2Seq pipeline for REBEL
	pipelineName := fmt.Sprintf("rebel:%s", filepath.Base(modelPath))
	pipelineOptions := []khugot.Seq2SeqOption{
		pipelines.WithSeq2SeqMaxTokens(config.MaxLength),
		pipelines.WithNumReturnSequences(1),
	}

	pipelineConfig := khugot.Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      pipelineName,
		Options:   pipelineOptions,
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		if !sessionShared {
			session.Destroy()
		}
		logger.Error("Failed to create REBEL pipeline", zap.Error(err))
		return nil, fmt.Errorf("creating REBEL pipeline: %w", err)
	}

	logger.Info("REBEL model initialization complete",
		zap.String("model_id", config.ModelID),
		zap.Int("max_length", config.MaxLength))

	return &HugotREBEL{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: sessionShared,
		config:        config,
	}, nil
}

// NewHugotREBELWithSessionManager creates a new REBEL model using a SessionManager.
func NewHugotREBELWithSessionManager(modelPath string, sessionManager *hugot.SessionManager, logger *zap.Logger) (*HugotREBEL, hugot.BackendType, error) {
	if modelPath == "" {
		return nil, "", errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	if sessionManager == nil {
		model, err := NewHugotREBELWithSession(modelPath, nil, logger)
		return model, hugot.BackendGo, err
	}

	logger.Info("Initializing Hugot REBEL model with SessionManager",
		zap.String("modelPath", modelPath))

	// Load REBEL config
	config := DefaultConfig()
	configPath := filepath.Join(modelPath, "rebel_config.json")
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			logger.Warn("Failed to parse REBEL config", zap.Error(err))
		}
	}

	// Get session from SessionManager
	session, backendUsed, err := sessionManager.GetSessionForModel(nil)
	if err != nil {
		logger.Error("Failed to get session from SessionManager", zap.Error(err))
		return nil, backendUsed, fmt.Errorf("getting session from SessionManager: %w", err)
	}

	// Create Seq2Seq pipeline for REBEL
	pipelineName := fmt.Sprintf("rebel:%s", filepath.Base(modelPath))
	pipelineOptions := []khugot.Seq2SeqOption{
		pipelines.WithSeq2SeqMaxTokens(config.MaxLength),
		pipelines.WithNumReturnSequences(1),
	}

	pipelineConfig := khugot.Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      pipelineName,
		Options:   pipelineOptions,
	}

	pipeline, err := khugot.NewPipeline(session, pipelineConfig)
	if err != nil {
		logger.Error("Failed to create REBEL pipeline", zap.Error(err))
		return nil, backendUsed, fmt.Errorf("creating REBEL pipeline: %w", err)
	}

	logger.Info("REBEL model initialization complete",
		zap.String("model_id", config.ModelID),
		zap.Int("max_length", config.MaxLength))

	return &HugotREBEL{
		session:       session,
		pipeline:      pipeline,
		logger:        logger,
		sessionShared: true,
		config:        config,
	}, backendUsed, nil
}

// --- ner.Model interface ---

// Recognize extracts entities from the given texts.
// REBEL is primarily a relation extractor, so this returns the entities
// extracted as subjects and objects from relation triplets.
func (h *HugotREBEL) Recognize(ctx context.Context, texts []string) ([][]ner.Entity, error) {
	entities, _, err := h.ExtractRelations(ctx, texts, nil, nil)
	return entities, err
}

// Close releases resources.
func (h *HugotREBEL) Close() error {
	var errs []error

	if h.pipeline != nil {
		if err := h.pipeline.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("destroying pipeline: %w", err))
		}
	}

	if h.session != nil && !h.sessionShared {
		h.logger.Info("Destroying Hugot session (owned by this REBEL model)")
		h.session.Destroy()
	}

	return errors.Join(errs...)
}

// --- ner.Recognizer interface ---

// RecognizeWithLabels extracts entities of the specified types.
// REBEL doesn't support custom entity labels - it extracts whatever entities
// appear in relation triplets. The labels parameter is ignored.
func (h *HugotREBEL) RecognizeWithLabels(ctx context.Context, texts []string, labels []string) ([][]ner.Entity, error) {
	return h.Recognize(ctx, texts)
}

// Labels returns the default entity labels this model uses.
// REBEL doesn't have predefined entity labels - it extracts entities from relations.
func (h *HugotREBEL) Labels() []string {
	return []string{} // REBEL extracts entities dynamically from relations
}

// --- Relation extraction methods (ner.Recognizer) ---

// ExtractRelations extracts relation triplets from the given texts.
// Returns entities (subjects and objects) and relations between them.
// The entityLabels and relationLabels parameters are ignored by REBEL.
func (h *HugotREBEL) ExtractRelations(ctx context.Context, texts []string, entityLabels []string, relationLabels []string) ([][]ner.Entity, [][]ner.Relation, error) {
	if len(texts) == 0 {
		return [][]ner.Entity{}, [][]ner.Relation{}, nil
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	default:
	}

	h.logger.Debug("Starting REBEL relation extraction",
		zap.Int("num_inputs", len(texts)))

	// Run the pipeline
	output, err := h.pipeline.RunPipeline(texts)
	if err != nil {
		h.logger.Error("REBEL generation failed", zap.Error(err))
		return nil, nil, fmt.Errorf("running REBEL pipeline: %w", err)
	}

	// Parse triplets from generated text and convert to ner.Entity/ner.Relation
	allEntities := make([][]ner.Entity, len(texts))
	allRelations := make([][]ner.Relation, len(texts))

	for i, generatedTexts := range output.GeneratedTexts {
		if len(generatedTexts) > 0 {
			rawOutput := generatedTexts[0]
			triplets := h.parseREBELOutput(rawOutput)

			// Convert triplets to entities and relations
			entities, relations := h.tripletsToNER(texts[i], triplets)
			allEntities[i] = entities
			allRelations[i] = relations

			h.logger.Debug("REBEL extraction completed",
				zap.Int("text_index", i),
				zap.Int("triplets_parsed", len(triplets)),
				zap.Int("entities", len(entities)),
				zap.Int("relations", len(relations)))
		} else {
			allEntities[i] = []ner.Entity{}
			allRelations[i] = []ner.Relation{}
			h.logger.Warn("REBEL generated no output for text", zap.Int("text_index", i))
		}
	}

	h.logger.Info("REBEL relation extraction completed",
		zap.Int("num_inputs", len(texts)))

	return allEntities, allRelations, nil
}

// ExtractAnswers performs extractive question answering.
// REBEL does not support question answering - use GLiNER multitask models instead.
func (h *HugotREBEL) ExtractAnswers(ctx context.Context, questions []string, contexts []string) ([]ner.Answer, error) {
	return nil, ner.ErrNotSupported
}

// RelationLabels returns the default relation labels this model uses.
// REBEL extracts relations dynamically - it doesn't have a fixed set of labels.
func (h *HugotREBEL) RelationLabels() []string {
	return []string{} // REBEL extracts relation types dynamically
}

// --- Helper methods ---

// Config returns the REBEL configuration.
func (h *HugotREBEL) Config() Config {
	return h.config
}

// triplet represents an extracted relation triplet from REBEL output.
type triplet struct {
	Subject  string
	Object   string
	Relation string
	Score    float32
}

// tripletsToNER converts REBEL triplets to NER entities and relations.
func (h *HugotREBEL) tripletsToNER(text string, triplets []triplet) ([]ner.Entity, []ner.Relation) {
	// Track unique entities to avoid duplicates
	entityMap := make(map[string]ner.Entity)
	var relations []ner.Relation

	for _, t := range triplets {
		// Find or create subject entity
		subjectKey := t.Subject
		if _, exists := entityMap[subjectKey]; !exists {
			start, end := findSpan(text, t.Subject)
			entityMap[subjectKey] = ner.Entity{
				Text:  t.Subject,
				Label: "ENTITY", // REBEL doesn't provide entity types
				Start: start,
				End:   end,
				Score: t.Score,
			}
		}

		// Find or create object entity
		objectKey := t.Object
		if _, exists := entityMap[objectKey]; !exists {
			start, end := findSpan(text, t.Object)
			entityMap[objectKey] = ner.Entity{
				Text:  t.Object,
				Label: "ENTITY",
				Start: start,
				End:   end,
				Score: t.Score,
			}
		}

		// Create relation
		relations = append(relations, ner.Relation{
			HeadEntity: entityMap[subjectKey],
			TailEntity: entityMap[objectKey],
			Label:      t.Relation,
			Score:      t.Score,
		})
	}

	// Convert entity map to slice
	entities := make([]ner.Entity, 0, len(entityMap))
	for _, e := range entityMap {
		entities = append(entities, e)
	}

	return entities, relations
}

// findSpan finds the character offsets of a substring in text.
// Returns -1, -1 if not found.
func findSpan(text, substring string) (int, int) {
	idx := strings.Index(strings.ToLower(text), strings.ToLower(substring))
	if idx == -1 {
		return -1, -1
	}
	return idx, idx + len(substring)
}

// parseREBELOutput parses REBEL's generated text into structured triplets.
//
// REBEL output format with special tokens:
// <s><triplet> Subject <subj> Object <obj> relation <triplet> Subject2 <subj> Object2 <obj> relation2 </s>
//
// Fallback format (when tokenizer strips special tokens):
// Subject  Object  relation  Subject2  Object2  relation2
// (elements separated by double spaces)
func (h *HugotREBEL) parseREBELOutput(text string) []triplet {
	var triplets []triplet

	// Remove start/end tokens
	text = strings.ReplaceAll(text, "<s>", "")
	text = strings.ReplaceAll(text, "</s>", "")
	text = strings.ReplaceAll(text, "<pad>", "")
	text = strings.TrimSpace(text)

	// Check if special tokens are present
	hasSpecialTokens := strings.Contains(text, h.config.TripletToken) ||
		strings.Contains(text, h.config.SubjectToken) ||
		strings.Contains(text, h.config.ObjectToken)

	if hasSpecialTokens {
		// Parse using special tokens
		parts := strings.Split(text, h.config.TripletToken)
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			if t := h.parseTripletPart(part); t != nil {
				triplets = append(triplets, *t)
			}
		}
	} else {
		// Fallback: parse by double-space separation
		triplets = h.parseREBELOutputNoTokens(text)
	}

	return triplets
}

// parseREBELOutputNoTokens parses REBEL output when special tokens are stripped.
// The format is elements separated by double spaces: "Subject  Object  relation  ..."
func (h *HugotREBEL) parseREBELOutputNoTokens(text string) []triplet {
	var triplets []triplet

	// Split by double space
	parts := strings.Split(text, "  ")

	// Filter out empty parts
	var elements []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			elements = append(elements, p)
		}
	}

	// Each triplet consists of 3 elements: subject, object, relation
	for i := 0; i+2 < len(elements); i += 3 {
		subject := elements[i]
		object := elements[i+1]
		relation := elements[i+2]

		if subject != "" && object != "" && relation != "" {
			triplets = append(triplets, triplet{
				Subject:  subject,
				Object:   object,
				Relation: relation,
			})
		}
	}

	return triplets
}

// parseTripletPart parses a single triplet from REBEL output.
func (h *HugotREBEL) parseTripletPart(part string) *triplet {
	subjToken := h.config.SubjectToken
	objToken := h.config.ObjectToken

	if !strings.Contains(part, subjToken) || !strings.Contains(part, objToken) {
		return nil
	}

	// Split by <subj> first
	subjSplit := strings.SplitN(part, subjToken, 2)
	if len(subjSplit) != 2 {
		return nil
	}
	subject := strings.TrimSpace(subjSplit[0])

	// The rest contains object and relation
	rest := subjSplit[1]

	// Split by <obj>
	objSplit := strings.SplitN(rest, objToken, 2)
	if len(objSplit) != 2 {
		return nil
	}
	object := strings.TrimSpace(objSplit[0])
	relation := strings.TrimSpace(objSplit[1])

	if subject == "" || object == "" || relation == "" {
		return nil
	}

	return &triplet{
		Subject:  subject,
		Object:   object,
		Relation: relation,
	}
}

// IsREBELModel checks if the model path contains a REBEL model.
// It looks for rebel_config.json or encoder/decoder ONNX files typical of REBEL.
func IsREBELModel(modelPath string) bool {
	// Check for rebel_config.json
	configPath := filepath.Join(modelPath, "rebel_config.json")
	if _, err := os.Stat(configPath); err == nil {
		return true
	}

	// Check if model name contains "rebel"
	modelName := strings.ToLower(filepath.Base(modelPath))
	if strings.Contains(modelName, "rebel") {
		// Verify it has the expected files
		encoderPath := filepath.Join(modelPath, "encoder_model.onnx")
		decoderPath := filepath.Join(modelPath, "decoder_model.onnx")
		if _, err := os.Stat(encoderPath); err == nil {
			if _, err := os.Stat(decoderPath); err == nil {
				return true
			}
		}
	}

	return false
}
