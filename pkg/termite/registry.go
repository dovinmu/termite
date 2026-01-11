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

package termite

// ModelRegistry contains the catalog of supported models
var ModelRegistry = RegistryResponse{
	Models:              registryModels,
	Types:               modelTypes,
	QuantizationOptions: quantizationOptions,
}

var modelTypes = []ModelTypeInfo{
	{
		Type:        ModelTypeEmbedder,
		Name:        "Embedder",
		Description: "Generate vector embeddings from text or images for semantic search and similarity",
		Icon:        "Fingerprint",
	},
	{
		Type:        ModelTypeReranker,
		Name:        "Reranker",
		Description: "Re-rank documents by relevance to a query for improved search results",
		Icon:        "ArrowUpDown",
	},
	{
		Type:        ModelTypeChunker,
		Name:        "Chunker",
		Description: "Semantic text chunking and segmentation for document processing",
		Icon:        "Scissors",
	},
	{
		Type:        ModelTypeRecognizer,
		Name:        "Recognizer",
		Description: "Entity recognition, relation extraction, and question answering",
		Icon:        "Tag",
	},
	{
		Type:        ModelTypeRewriter,
		Name:        "Rewriter",
		Description: "Sequence-to-sequence text transformation like paraphrasing and question generation",
		Icon:        "RefreshCw",
	},
	{
		Type:        ModelTypeGenerator,
		Name:        "Generator",
		Description: "Generative language models for text generation and function calling",
		Icon:        "Sparkles",
	},
}

var quantizationOptions = []QuantizationOption{
	{
		Type:        QuantizationTypeF32,
		Name:        "Float32",
		Description: "Full precision - largest size, highest accuracy",
	},
	{
		Type:        QuantizationTypeF16,
		Name:        "Float16",
		Description: "Half precision - recommended for ARM64/M-series Macs",
		Recommended: true,
	},
	{
		Type:        QuantizationTypeBf16,
		Name:        "BFloat16",
		Description: "Brain floating point - balanced precision for training and inference",
	},
	{
		Type:        QuantizationTypeI8,
		Name:        "INT8",
		Description: "8-bit integer quantization - smallest size, fastest inference",
	},
	{
		Type:        QuantizationTypeI8St,
		Name:        "INT8 Static",
		Description: "Static INT8 quantization - calibrated for specific data distributions",
	},
	{
		Type:          QuantizationTypeI4,
		Name:          "INT4",
		Description:   "4-bit integer quantization - very small, for generators only",
		GeneratorOnly: true,
	},
	{
		Type:          QuantizationTypeI4Cuda,
		Name:          "INT4 CUDA",
		Description:   "CUDA-optimized 4-bit quantization - for NVIDIA GPUs, generators only",
		GeneratorOnly: true,
	},
}

var registryModels = []RegistryModel{
	// ==================== EMBEDDERS ====================
	{
		Id:          "bge-small-en-v1.5",
		Name:        "BGE Small English v1.5",
		Source:      "BAAI/bge-small-en-v1.5",
		SourceUrl:   "https://huggingface.co/BAAI/bge-small-en-v1.5",
		Type:        ModelTypeEmbedder,
		Description: "Compact and efficient English sentence embedding model. Great balance of speed and quality for semantic search applications.",
		Variants:    []QuantizationType{QuantizationTypeF16, QuantizationTypeI8},
		Backends:    []Backend{BackendOnnx, BackendXla, BackendGo},
		InRegistry:  true,
	},
	{
		Id:          "clip-vit-base-patch32",
		Name:        "CLIP ViT-Base Patch32",
		Source:      "openai/clip-vit-base-patch32",
		SourceUrl:   "https://huggingface.co/openai/clip-vit-base-patch32",
		Type:        ModelTypeEmbedder,
		Description: "Multimodal model that creates joint embeddings for both text and images. Enables cross-modal search and similarity. Requires ONNX Runtime.",
		Variants:    []QuantizationType{QuantizationTypeF16, QuantizationTypeI8},
		Backends:    []Backend{BackendOnnx},
		InRegistry:  true,
	},
	{
		Id:          "all-minilm-l6-v2",
		Name:        "All-MiniLM-L6-v2",
		Source:      "sentence-transformers/all-MiniLM-L6-v2",
		SourceUrl:   "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
		Type:        ModelTypeEmbedder,
		Description: "Extremely fast sentence embeddings model. Optimized for speed while maintaining good quality. Ideal for high-throughput applications.",
		Variants:    []QuantizationType{QuantizationTypeF32, QuantizationTypeF16, QuantizationTypeI8},
		Backends:    []Backend{BackendOnnx},
		InRegistry:  false,
	},
	{
		Id:          "all-mpnet-base-v2",
		Name:        "All-MPNet-Base-v2",
		Source:      "sentence-transformers/all-mpnet-base-v2",
		SourceUrl:   "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
		Type:        ModelTypeEmbedder,
		Description: "High-quality sentence embeddings with excellent semantic understanding. Best accuracy among sentence-transformers models.",
		Variants:    []QuantizationType{QuantizationTypeF32, QuantizationTypeF16, QuantizationTypeI8},
		Backends:    []Backend{BackendOnnx},
		InRegistry:  false,
	},

	// ==================== RERANKERS ====================
	{
		Id:          "mxbai-rerank-base-v1",
		Name:        "MixedBread Rerank Base v1",
		Source:      "mixedbread-ai/mxbai-rerank-base-v1",
		SourceUrl:   "https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1",
		Type:        ModelTypeReranker,
		Description: "State-of-the-art document reranking model. Significantly improves search relevance by rescoring candidate documents.",
		Variants:    []QuantizationType{QuantizationTypeF16, QuantizationTypeI8},
		Backends:    []Backend{BackendOnnx},
		InRegistry:  true,
	},

	// ==================== CHUNKERS ====================
	{
		Id:          "chonky-mmbert-small-multilingual-1",
		Name:        "Chonky mMBERT Small Multilingual",
		Source:      "mirth/chonky_mmbert_small_multilingual_1",
		SourceUrl:   "https://huggingface.co/mirth/chonky_mmbert_small_multilingual_1",
		Type:        ModelTypeChunker,
		Description: "Neural semantic chunking model with multilingual support. Intelligently segments documents at natural semantic boundaries.",
		Variants:    []QuantizationType{QuantizationTypeF16, QuantizationTypeI8},
		Backends:    []Backend{BackendOnnx},
		InRegistry:  true,
	},

	// ==================== RECOGNIZERS ====================
	{
		Id:           "rebel-large",
		Name:         "REBEL Large",
		Source:       "Babelscape/rebel-large",
		SourceUrl:    "https://huggingface.co/Babelscape/rebel-large",
		Type:         ModelTypeRecognizer,
		Description:  "Relation extraction model based on BART. Extracts entity-relation triplets from text for knowledge graph construction.",
		Capabilities: []RecognizerCapability{RecognizerCapabilityRelations},
		Architecture: "BART Encoder-Decoder",
		Variants:     []QuantizationType{QuantizationTypeF16, QuantizationTypeI8},
		Backends:     []Backend{BackendOnnx},
		InRegistry:   true,
	},
	{
		Id:           "bert-base-ner",
		Name:         "BERT Base NER",
		Source:       "dslim/bert-base-NER",
		SourceUrl:    "https://huggingface.co/dslim/bert-base-NER",
		Type:         ModelTypeRecognizer,
		Description:  "Standard BERT-based named entity recognition. Identifies Person, Organization, Location, and Miscellaneous entities.",
		Capabilities: []RecognizerCapability{RecognizerCapabilityLabels},
		Architecture: "BERT",
		Variants:     []QuantizationType{QuantizationTypeF32, QuantizationTypeF16, QuantizationTypeI8},
		Backends:     []Backend{BackendOnnx},
		InRegistry:   false,
	},
	{
		Id:           "bert-large-ner",
		Name:         "BERT Large NER",
		Source:       "dslim/bert-large-NER",
		SourceUrl:    "https://huggingface.co/dslim/bert-large-NER",
		Type:         ModelTypeRecognizer,
		Description:  "Larger BERT model for named entity recognition. Higher accuracy than base model for complex entity extraction tasks.",
		Capabilities: []RecognizerCapability{RecognizerCapabilityLabels},
		Architecture: "BERT Large",
		Variants:     []QuantizationType{QuantizationTypeF32, QuantizationTypeF16, QuantizationTypeI8},
		Backends:     []Backend{BackendOnnx},
		InRegistry:   false,
	},
	{
		Id:           "gliner-small-v2.1",
		Name:         "GLiNER Small v2.1",
		Source:       "urchade/gliner_small-v2.1",
		SourceUrl:    "https://huggingface.co/urchade/gliner_small-v2.1",
		Type:         ModelTypeRecognizer,
		Description:  "Zero-shot NER model that can extract entities for any label without retraining. Specify custom entity types at inference time.",
		Capabilities: []RecognizerCapability{RecognizerCapabilityLabels, RecognizerCapabilityZeroshot},
		Architecture: "GLiNER",
		Variants:     []QuantizationType{QuantizationTypeF32, QuantizationTypeF16, QuantizationTypeI8},
		Backends:     []Backend{BackendOnnx},
		InRegistry:   false,
	},
	{
		Id:           "gliner-multitask-large-v0.5",
		Name:         "GLiNER Multitask Large v0.5",
		Source:       "knowledgator/gliner-multitask-large-v0.5",
		SourceUrl:    "https://huggingface.co/knowledgator/gliner-multitask-large-v0.5",
		Type:         ModelTypeRecognizer,
		Description:  "Powerful multitask extraction model. Combines NER, zero-shot labeling, relation extraction, and question answering in one model.",
		Capabilities: []RecognizerCapability{RecognizerCapabilityLabels, RecognizerCapabilityZeroshot, RecognizerCapabilityRelations, RecognizerCapabilityAnswers},
		Architecture: "GLiNER Multitask",
		Variants:     []QuantizationType{QuantizationTypeF32, QuantizationTypeF16, QuantizationTypeI8},
		Backends:     []Backend{BackendOnnx},
		InRegistry:   false,
	},

	// ==================== REWRITERS ====================
	{
		Id:           "flan-t5-small-squad-qg",
		Name:         "FLAN-T5 Small Question Generator",
		Source:       "lmqg/flan-t5-small-squad-qg",
		SourceUrl:    "https://huggingface.co/lmqg/flan-t5-small-squad-qg",
		Type:         ModelTypeRewriter,
		Description:  "Question generation model fine-tuned on SQuAD. Generates relevant questions from text passages for training data creation.",
		Architecture: "T5 Encoder-Decoder",
		Variants:     []QuantizationType{QuantizationTypeF16, QuantizationTypeI8},
		Backends:     []Backend{BackendOnnx},
		InRegistry:   true,
	},
	{
		Id:           "pegasus-paraphrase",
		Name:         "PEGASUS Paraphrase",
		Source:       "tuner007/pegasus_paraphrase",
		SourceUrl:    "https://huggingface.co/tuner007/pegasus_paraphrase",
		Type:         ModelTypeRewriter,
		Description:  "Paraphrasing model based on PEGASUS. Rewrites text while preserving meaning for data augmentation and style transfer.",
		Architecture: "PEGASUS Encoder-Decoder",
		Variants:     []QuantizationType{QuantizationTypeF16, QuantizationTypeI8},
		Backends:     []Backend{BackendOnnx},
		InRegistry:   true,
	},

	// ==================== GENERATORS ====================
	{
		Id:          "functiongemma-270m-it",
		Name:        "FunctionGemma 270M Instruct",
		Source:      "google/functiongemma-270m-it",
		SourceUrl:   "https://huggingface.co/google/functiongemma-270m-it",
		Type:        ModelTypeGenerator,
		Description: "Compact instruction-tuned model optimized for function calling. Ideal for tool use and structured output generation.",
		Size:        "270M parameters",
		Variants:    []QuantizationType{QuantizationTypeF16, QuantizationTypeI8, QuantizationTypeI4, QuantizationTypeI4Cuda},
		Backends:    []Backend{BackendOnnx},
		InRegistry:  true,
	},
	{
		Id:          "gemma-3-1b-it",
		Name:        "Gemma 3 1B Instruct",
		Source:      "google/gemma-3-1b-it",
		SourceUrl:   "https://huggingface.co/google/gemma-3-1b-it",
		Type:        ModelTypeGenerator,
		Description: "Larger instruction-tuned Gemma model. Better reasoning and generation quality for more complex tasks.",
		Size:        "1B parameters",
		Variants:    []QuantizationType{QuantizationTypeF16, QuantizationTypeI8, QuantizationTypeI4, QuantizationTypeI4Cuda},
		Backends:    []Backend{BackendOnnx},
		InRegistry:  false,
	},
}
