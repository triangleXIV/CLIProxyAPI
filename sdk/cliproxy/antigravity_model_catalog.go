package cliproxy

import (
	"encoding/json"
	"slices"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
)

type antigravityDiscoveredModel struct {
	DisplayName               string          `json:"displayName"`
	IsInternal                bool            `json:"isInternal"`
	MaxTokens                 int             `json:"maxTokens"`
	MaxOutputTokens           int             `json:"maxOutputTokens"`
	SupportsImages            bool            `json:"supportsImages"`
	SupportsVideo             bool            `json:"supportsVideo"`
	SupportedMimeTypes        map[string]bool `json:"supportedMimeTypes"`
	SupportedOutputModalities []string        `json:"supportedOutputModalities"`
	SupportsThinking          bool            `json:"supportsThinking"`
	MinThinkingBudget         *int            `json:"minThinkingBudget"`
	ThinkingBudget            *int            `json:"thinkingBudget"`
}

func isRetiredAntigravityModel(id string) bool {
	// These catalog entries now return a retirement notice with HTTP 200 rather
	// than generating a reply. Catalog presence alone does not make them callable.
	switch id {
	case "gemini-3-flash-agent", "gemini-3.5-flash-extra-low", "gemini-3.5-flash-low":
		return true
	default:
		return false
	}
}

// Keep curated definitions for existing models, but discover eligible additions
// from the same account catalog that grants their routing entitlement.
func appendAntigravityDiscoveredModels(models []*ModelInfo, hints antigravityModelCapabilityHints) []*ModelInfo {
	seen := make(map[string]struct{}, len(models))
	for _, model := range models {
		seen[normalizeAntigravityFetchedModelID(model.ID)] = struct{}{}
	}
	ids := make([]string, 0, len(hints.Models))
	for id := range hints.Models {
		if _, entitled := hints.ModelIDs[id]; !entitled {
			continue
		}
		if _, exists := seen[id]; !exists {
			ids = append(ids, id)
		}
	}
	slices.Sort(ids)
	for _, id := range ids {
		if model := discoveredAntigravityModelInfo(id, hints.Models[id], hints.ImageGenerationModelIDs); model != nil {
			models = append(models, model)
		}
	}
	return models
}

func discoveredAntigravityModelInfo(id string, raw json.RawMessage, images map[string]struct{}) *ModelInfo {
	if isRetiredAntigravityModel(id) {
		return nil
	}
	var discovered antigravityDiscoveredModel
	if json.Unmarshal(raw, &discovered) != nil || discovered.IsInternal || strings.TrimSpace(discovered.DisplayName) == "" {
		return nil
	}
	// The executor supports these conversational families. Editor completions,
	// tier routers, transcription and other media protocols are not chat models.
	isGemini := strings.HasPrefix(id, "gemini-")
	if !isGemini && !strings.HasPrefix(id, "claude-") && !strings.HasPrefix(id, "gpt-oss-") {
		return nil
	}
	if strings.HasSuffix(id, "-tiered") || strings.ContainsAny(id, " /\\()") {
		return nil
	}
	for _, marker := range []string{"-tts", "-audio", "-live", "-video", "-veo", "-transcription"} {
		if strings.Contains(id, marker) {
			return nil
		}
	}
	_, imageGeneration := images[id]
	if imageGeneration && (!isGemini || !strings.Contains(id, "-image")) {
		return nil
	}
	if strings.Contains(id, "-image") && !imageGeneration {
		return nil
	}
	for _, modality := range discovered.SupportedOutputModalities {
		switch strings.ToLower(modality) {
		case "text":
		case "image":
			if !imageGeneration {
				return nil
			}
		default:
			return nil
		}
	}
	info := &ModelInfo{
		ID: id, Object: "model", OwnedBy: "antigravity", Type: "antigravity",
		Name: id, DisplayName: strings.TrimSpace(discovered.DisplayName),
		Description:   strings.TrimSpace(discovered.DisplayName),
		ContextLength: max(0, discovered.MaxTokens), MaxCompletionTokens: max(0, discovered.MaxOutputTokens),
		SupportedInputModalities: []string{"text"}, SupportedOutputModalities: []string{"text"},
	}
	inputImage, inputAudio, inputVideo := discovered.SupportsImages || imageGeneration, false, discovered.SupportsVideo
	for mime, supported := range discovered.SupportedMimeTypes {
		if !supported {
			continue
		}
		switch {
		case strings.HasPrefix(mime, "image/"):
			inputImage = true
		case strings.HasPrefix(mime, "audio/"):
			inputAudio = true
		case strings.HasPrefix(mime, "video/"):
			inputVideo = true
		}
	}
	if inputImage {
		info.SupportedInputModalities = append(info.SupportedInputModalities, "image")
	}
	if inputAudio {
		info.SupportedInputModalities = append(info.SupportedInputModalities, "audio")
	}
	if inputVideo {
		info.SupportedInputModalities = append(info.SupportedInputModalities, "video")
	}
	if imageGeneration {
		info.SupportedOutputModalities = append(info.SupportedOutputModalities, "image")
	}
	if discovered.SupportsThinking && discovered.MaxOutputTokens > 1 {
		minimum := 1
		if discovered.MinThinkingBudget != nil {
			minimum = max(0, *discovered.MinThinkingBudget)
		}
		maximum := discovered.MaxOutputTokens - 1
		if minimum <= maximum {
			info.Thinking = &registry.ThinkingSupport{
				Min: minimum, Max: maximum,
				ZeroAllowed:    discovered.MinThinkingBudget != nil && minimum == 0,
				DynamicAllowed: discovered.ThinkingBudget != nil && *discovered.ThinkingBudget == -1,
			}
		}
	}
	return info
}
