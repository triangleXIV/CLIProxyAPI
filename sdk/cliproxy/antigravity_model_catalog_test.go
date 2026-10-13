package cliproxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
)

func TestAntigravityDiscoveredCatalogEligibilityAndMetadata(t *testing.T) {
	hints, ok := parseAntigravityModelCapabilityHints([]byte(`{
		"models": {
			"gemini-3.8-flash-low": {"displayName":"Gemini 3.8 Flash (Low)","maxTokens":1048576,"maxOutputTokens":65536,"supportsImages":true,"supportsVideo":true,"supportedMimeTypes":{"audio/wav":true},"supportsThinking":true,"minThinkingBudget":0,"thinkingBudget":1000},
			"gemini-3.8-flash-medium": {"displayName":"Gemini 3.8 Flash (Medium)","maxTokens":1048576,"maxOutputTokens":65536},
			"gemini-3-flash-agent": {"displayName":"Gemini 3.5 Flash (High)"},
			"gemini-3.5-flash-extra-low": {"displayName":"Gemini 3.5 Flash (Low)"},
			"gemini-3.5-flash-low": {"displayName":"Gemini 3.5 Flash (Medium)"},
			"gemini-2.5-pro": {"displayName":"Gemini 2.5 Pro"},
			"gemini-2.5-flash": {"displayName":"Gemini 3.5 Flash Lite"},
			"gemini-2.5-flash-lite": {"displayName":"Gemini 3.5 Flash Lite"},
			"gemini-2.5-flash-thinking": {"displayName":"Gemini 3.5 Flash Lite"},
			"gemini-3.1-flash-lite": {"displayName":"wrong upstream metadata","maxTokens":1},
			"gemini-3.8-flash-tiered": {"displayName":"Tier router"},
			"chat_20706": {"displayName":"Internal","isInternal":true},
			"tab_flash_lite_preview": {"displayName":"Editor"},
			"gemini-private": {"displayName":"Internal","isInternal":true},
			"gemini-malformed": [],
			"not-in-static": {},
			"gemini-tts": {"displayName":"Speech"},
			"gemini-media": {"displayName":"Speech","supportedOutputModalities":["AUDIO"]},
			"gemini-image-input": {"displayName":"Image input only"},
			"gemini-new-image": {"displayName":"New image"},
			"imagen-4": {"displayName":"Imagen"},
			"veo-3": {"displayName":"Veo"},
			"models/proactive-observer-v10": {"displayName":"Transcription"}
		},
		"imageGenerationModelIds":["gemini-new-image","imagen-4"],
		"webSearchModelIds":["gemini-3.8-flash-low"]
	}`))
	if !ok {
		t.Fatal("catalog parse failed")
	}
	models := applyAntigravityFetchedModelCapabilities(filterAntigravityModels(registry.GetAntigravityModels(), hints), hints)
	if len(models) != 8 {
		t.Fatalf("expected seven discovered models and one predefined model, got %v", models)
	}
	byID := make(map[string]*ModelInfo)
	for _, model := range models {
		byID[model.ID] = model
	}
	low := byID["gemini-3.8-flash-low"]
	if low == nil || low.ContextLength != 1048576 || low.MaxCompletionTokens != 65536 || !low.SupportsWebSearch {
		t.Fatalf("lost discovered metadata: %+v", low)
	}
	if !reflect.DeepEqual(low.SupportedInputModalities, []string{"text", "image", "audio", "video"}) || !reflect.DeepEqual(low.SupportedOutputModalities, []string{"text"}) {
		t.Fatalf("input media became generated media: %+v", low)
	}
	if low.Thinking == nil || low.Thinking.Max != 65535 || !low.Thinking.ZeroAllowed || low.Thinking.DynamicAllowed {
		t.Fatalf("incorrect thinking capability: %+v", low.Thinking)
	}
	if image := byID["gemini-new-image"]; image == nil || !reflect.DeepEqual(image.SupportedOutputModalities, []string{"text", "image"}) {
		t.Fatalf("lost image generation capability: %+v", image)
	}
	if known := byID["gemini-3.1-flash-lite"]; known == nil || known.ContextLength <= 1 || known.DisplayName == "wrong upstream metadata" {
		t.Fatalf("predefined metadata overwritten: %+v", known)
	}
	for _, id := range []string{"gemini-3-flash-agent", "gemini-3.5-flash-extra-low", "gemini-3.5-flash-low", "gemini-3.8-flash-tiered", "gemini-private", "gemini-malformed", "gemini-tts", "gemini-media", "chat_20706", "tab_flash_lite_preview", "imagen-4", "veo-3", "models/proactive-observer-v10", "gemini-image-input", "not-in-static"} {
		if byID[id] != nil {
			t.Fatalf("published unsupported model %s", id)
		}
	}
}

func TestAntigravityDiscoveredHintsCloneIsolation(t *testing.T) {
	hints, _ := parseAntigravityModelCapabilityHints([]byte(`{"models":{"gemini-new-image":{"displayName":"Image"}},"imageGenerationModelIds":["gemini-new-image"]}`))
	clone := hints.clone()
	clone.Models["gemini-new-image"][0] = '['
	delete(clone.ImageGenerationModelIDs, "gemini-new-image")
	if !json.Valid(hints.Models["gemini-new-image"]) || len(hints.ImageGenerationModelIDs) != 1 {
		t.Fatal("cache snapshot mutated through clone")
	}
	delete(hints.ModelIDs, "gemini-new-image")
	if models := filterAntigravityModels(nil, hints); len(models) != 0 {
		t.Fatal("unentitled metadata restored a model")
	}
}

func TestAntigravityRetiredModelsCannotReturnThroughCuratedCatalog(t *testing.T) {
	const id = "gemini-3-flash-agent"
	hints := antigravityModelCapabilityHints{ModelIDs: map[string]struct{}{id: {}}}
	if models := filterAntigravityModels([]*ModelInfo{{ID: id, DisplayName: "Gemini 3.5 Flash"}}, hints); len(models) != 0 {
		t.Fatal("curated catalog restored a retired model")
	}
}

func TestAntigravityDiscoveredAccountPublicationLifecycle(t *testing.T) {
	resetAntigravityCapabilityCache()
	t.Cleanup(resetAntigravityCapabilityCache)
	const model = "gemini-3.8-flash-low"
	var catalog atomic.Value
	catalog.Store(`{"models":{"gemini-3.8-flash-low":{"displayName":"Gemini Low","maxTokens":1048576,"maxOutputTokens":65536}}}`)
	var status atomic.Int32
	status.Store(http.StatusOK)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(int(status.Load()))
		_, _ = w.Write([]byte(catalog.Load().(string)))
	}))
	defer server.Close()
	svc := &Service{
		cfg:         &config.Config{OAuthModelAlias: map[string][]config.OAuthModelAlias{"antigravity": {{Name: model, Alias: "low-chat"}}}},
		coreManager: coreauth.NewManager(nil, nil, nil),
	}
	auth := antigravityTestAuth("discovered-account", server.URL)
	auth.Prefix = "tenant"
	auth, err := svc.coreManager.Register(t.Context(), auth)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { GlobalModelRegistry().UnregisterClient(auth.ID) })
	svc.registerModelsForAuth(t.Context(), auth)
	svc.WaitAntigravityProbes()
	assertPublished := func(want bool) {
		t.Helper()
		models := GlobalModelRegistry().GetModelsForClient(auth.ID)
		if !want {
			if len(models) != 0 {
				t.Fatalf("revoked model retained: %v", models)
			}
			return
		}
		if len(models) != 2 {
			t.Fatalf("alias and prefix missing: %v", models)
		}
		for _, info := range models {
			if info.ID != "low-chat" && info.ID != "tenant/low-chat" {
				t.Fatalf("unexpected published route %s", info.ID)
			}
		}
	}
	assertPublished(true)
	other := antigravityTestAuth("unentitled-account", server.URL)
	if models := svc.antigravityModelsForHints(other, antigravityModelCapabilityHints{ModelIDs: map[string]struct{}{}}); len(models) != 0 {
		t.Fatal("discovered model leaked to another account")
	}
	svc.cfg.OAuthExcludedModels = map[string][]string{"antigravity": {model}}
	if models := svc.antigravityModelsForHints(auth, svc.cachedAntigravityHints(auth)); len(models) != 0 {
		t.Fatal("discovery bypassed exclusion")
	}
	svc.cfg.OAuthExcludedModels = nil
	expire := func() {
		antigravityCapabilityMu.Lock()
		key := svc.antigravityCapabilityKey(auth)
		entry := antigravityCapabilityCache[key]
		entry.expiresAt = time.Time{}
		antigravityCapabilityCache[key] = entry
		antigravityAuthFailureCache = make(map[string]antigravityFailureState)
		antigravityCapabilityMu.Unlock()
	}
	expire()
	status.Store(http.StatusForbidden)
	svc.refreshAntigravityModels(t.Context())
	svc.WaitAntigravityProbes()
	assertPublished(true)
	expire()
	status.Store(http.StatusOK)
	catalog.Store(`{"models":{}}`)
	svc.refreshAntigravityModels(t.Context())
	svc.WaitAntigravityProbes()
	assertPublished(false)
}
