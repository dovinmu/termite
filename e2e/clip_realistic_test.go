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

package e2e

import (
	"context"
	"fmt"
	"image/color"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/client"
	"github.com/antflydb/termite/pkg/termite"
	"go.uber.org/zap/zaptest"
)

// TestCLIPRealisticSimilarity tests that CLIP produces meaningful cross-modal similarity.
//
// Working CLIP should produce:
// - Text-text similarity for similar concepts: 0.7-0.9
// - Text-text similarity for different concepts: 0.4-0.7
// - Cross-modal similarity for matching content: 0.20-0.40
// - Cross-modal similarity for non-matching: -0.05 to 0.15
//
// If cross-modal similarity is near zero for ALL pairs (matching and non-matching),
// the text and image embeddings are in different subspaces - likely a pooling bug.
func TestCLIPRealisticSimilarity(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	ensureRegistryModel(t, clipModelName, ModelTypeEmbedder)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	logger := zaptest.NewLogger(t)
	modelsDir := getTestModelsDir()
	port := findAvailablePort(t)
	serverURL := fmt.Sprintf("http://localhost:%d", port)

	config := termite.Config{
		ApiUrl:    serverURL,
		ModelsDir: modelsDir,
	}

	serverCtx, serverCancel := context.WithCancel(ctx)
	defer serverCancel()

	readyC := make(chan struct{})
	serverDone := make(chan struct{})

	go func() {
		defer close(serverDone)
		termite.RunAsTermite(serverCtx, logger, config, readyC)
	}()

	select {
	case <-readyC:
		t.Log("Server is ready")
	case <-time.After(60 * time.Second):
		t.Fatal("Timeout waiting for server to be ready")
	}

	termiteClient, err := client.NewTermiteClient(serverURL, nil)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	t.Run("TextTextSimilarity", func(t *testing.T) {
		// Test that text embeddings are semantically meaningful
		// Similar concepts should be more similar than different concepts
		pairs := []struct {
			text1, text2 string
			minSim       float64 // minimum expected similarity
			maxSim       float64 // maximum expected similarity
			desc         string
		}{
			// CLIP text encoder produces high similarity for related text.
			// These thresholds are calibrated for openai/clip-vit-base-patch32.
			{"a photo of a cat", "a cute kitten", 0.85, 0.98, "similar concepts"},
			{"a photo of a cat", "a photo of a dog", 0.85, 0.98, "related animals"},
			{"a photo of a cat", "a car on the highway", 0.65, 0.85, "unrelated concepts"},
		}

		for _, p := range pairs {
			embs, err := termiteClient.Embed(ctx, clipModelName, []string{p.text1, p.text2})
			if err != nil {
				t.Fatalf("Embed failed: %v", err)
			}

			sim := cosineSimilarity(embs[0], embs[1])
			t.Logf("'%s' vs '%s': %.4f (%s)", p.text1, p.text2, sim, p.desc)

			if sim < p.minSim {
				t.Errorf("Similarity %.4f < expected min %.4f for %s", sim, p.minSim, p.desc)
			}
			if sim > p.maxSim {
				t.Errorf("Similarity %.4f > expected max %.4f for %s", sim, p.maxSim, p.desc)
			}
		}

		// Key check: cat-kitten should be MORE similar than cat-car
		catKitten, _ := termiteClient.Embed(ctx, clipModelName, []string{"a photo of a cat", "a cute kitten"})
		catCar, _ := termiteClient.Embed(ctx, clipModelName, []string{"a photo of a cat", "a car"})

		simCatKitten := cosineSimilarity(catKitten[0], catKitten[1])
		simCatCar := cosineSimilarity(catCar[0], catCar[1])

		if simCatKitten <= simCatCar {
			t.Errorf("SEMANTIC BUG: cat-kitten (%.4f) should be > cat-car (%.4f)", simCatKitten, simCatCar)
		}
	})

	t.Run("CrossModalMagnitude", func(t *testing.T) {
		// The key test: cross-modal similarity should NOT be near zero
		// Even synthetic images should give reasonable similarity with matching text

		// Create red square image
		redSquareData := createTestImage(t, 100, 100, color.RGBA{255, 0, 0, 255})
		redEmb := embedImage(t, ctx, serverURL, clipModelName, redSquareData)

		// Get text embeddings
		textEmbs, err := termiteClient.Embed(ctx, clipModelName, []string{
			"a red square",
			"a red rectangle",
			"a blue circle",
			"a photograph of a golden retriever dog",
		})
		if err != nil {
			t.Fatalf("Embed failed: %v", err)
		}

		// Calculate all cross-modal similarities
		sims := make([]float64, len(textEmbs))
		labels := []string{"red square", "red rectangle", "blue circle", "dog photo"}
		for i, textEmb := range textEmbs {
			sims[i] = cosineSimilarity(textEmb, redEmb)
			t.Logf("'%s' vs red square image: %.4f", labels[i], sims[i])
		}

		// Critical check: if ALL similarities are near zero, embeddings are misaligned
		allNearZero := true
		for _, s := range sims {
			if s > 0.10 || s < -0.10 {
				allNearZero = false
				break
			}
		}

		if allNearZero {
			t.Errorf("CROSS-MODAL BUG: All similarities near zero (%.4f, %.4f, %.4f, %.4f). "+
				"Text and image embeddings appear to be in different subspaces. "+
				"This usually indicates wrong pooling strategy (should use EOS for CLIP, not mean).",
				sims[0], sims[1], sims[2], sims[3])
		}

		// "red square" should match better than "dog"
		if sims[0] <= sims[3] {
			t.Errorf("'red square' (%.4f) should match red square image better than 'dog' (%.4f)",
				sims[0], sims[3])
		}

		// Minimum expected cross-modal similarity for matching content
		// Working CLIP typically gives 0.15-0.30 even for synthetic images
		minExpected := 0.12
		if sims[0] < minExpected {
			t.Errorf("Cross-modal similarity for matching content (%.4f) below threshold (%.4f). "+
				"Expected 0.15-0.30 for working CLIP.", sims[0], minExpected)
		}
	})

	// Cleanup
	serverCancel()
	select {
	case <-serverDone:
	case <-time.After(30 * time.Second):
		t.Error("Timeout waiting for server shutdown")
	}
}
