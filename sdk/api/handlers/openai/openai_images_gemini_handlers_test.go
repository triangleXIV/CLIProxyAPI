package openai

import (
	"net/http"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
	"github.com/tidwall/gjson"
)

func TestIsGeminiImagesModel(t *testing.T) {
	for _, model := range []string{
		"gemini-3.1-flash-image",
		"gemini-2.5-flash-image",
		"gemini-2.5-flash-image-preview",
		"gemini-3-pro-image",
		"google/gemini-3.1-flash-image",
		"Gemini-3.1-Flash-Image",
	} {
		if !isGeminiImagesModel(model) {
			t.Fatalf("expected %s to be a Gemini images model", model)
		}
		if !isSupportedImagesModel(model) {
			t.Fatalf("expected %s to be supported on the images endpoints", model)
		}
	}
	for _, model := range []string{
		"gemini-3-flash",
		"gemini-3.1-pro-low",
		"gpt-image-2",
		"grok-imagine-image",
		"imagen-4.0-generate",
		"",
	} {
		if isGeminiImagesModel(model) {
			t.Fatalf("expected %q to be rejected as a Gemini images model", model)
		}
	}
}

func TestGeminiImagesAspectRatioFromSize(t *testing.T) {
	cases := map[string]string{
		"1024x1024": "1:1",
		"2048x2048": "1:1",
		"1536x1024": "3:2",
		"1024x1536": "2:3",
		"1792x1024": "3:2",
		"1024x1792": "2:3",
		"1920x1080": "16:9",
		"1080x1920": "9:16",
		"16:9":      "16:9",
		"9:16":      "9:16",
		"landscape": "16:9",
		"portrait":  "9:16",
		"square":    "1:1",
		"auto":      "",
		"":          "",
		"weird":     "",
	}
	for size, want := range cases {
		if got := geminiImagesAspectRatioFromSize(size); got != want {
			t.Fatalf("geminiImagesAspectRatioFromSize(%q) = %q, want %q", size, got, want)
		}
	}
}

func TestGeminiImagesSizeTier(t *testing.T) {
	cases := []struct {
		explicit string
		size     string
		quality  string
		want     string
	}{
		{explicit: "4K", want: "4K"},
		{explicit: "2k", want: "2K"},
		{explicit: "1K", want: "1K"},
		{size: "4096", want: "4K"},
		{size: "2048", want: "2K"},
		{size: "1024x1024", quality: "high", want: "4K"},
		{quality: "low", want: "1K"},
		{quality: "standard", want: "1K"},
		{quality: "medium", want: "2K"},
		{quality: "hd", want: "4K"},
		{quality: "auto", want: ""},
		{want: ""},
	}
	for _, tc := range cases {
		if got := geminiImagesSizeTier(tc.explicit, tc.size, tc.quality); got != tc.want {
			t.Fatalf("geminiImagesSizeTier(%q, %q, %q) = %q, want %q", tc.explicit, tc.size, tc.quality, got, tc.want)
		}
	}
}

func TestBuildGeminiImagesGenerationConfig(t *testing.T) {
	cfg := buildGeminiImagesGenerationConfig(geminiImagesOptions{
		AspectRatio: "16:9",
		ImageSize:   "2K",
	})
	if got := gjson.GetBytes(cfg, "imageConfig.aspectRatio").String(); got != "16:9" {
		t.Fatalf("aspectRatio = %q, want 16:9", got)
	}
	if got := gjson.GetBytes(cfg, "imageConfig.imageSize").String(); got != "2K" {
		t.Fatalf("imageSize = %q, want 2K", got)
	}
	if got := gjson.GetBytes(cfg, "responseModalities").Raw; got != `["IMAGE","TEXT"]` {
		t.Fatalf("responseModalities = %s, want [IMAGE,TEXT]", got)
	}
	if gjson.GetBytes(cfg, "candidateCount").Exists() {
		t.Fatal("candidateCount must be omitted for n<=1")
	}

	cfg = buildGeminiImagesGenerationConfig(geminiImagesOptions{Size: "1024x1536", Quality: "high", N: 3})
	if got := gjson.GetBytes(cfg, "imageConfig.aspectRatio").String(); got != "2:3" {
		t.Fatalf("size 1024x1536 aspectRatio = %q, want 2:3", got)
	}
	if got := gjson.GetBytes(cfg, "imageConfig.imageSize").String(); got != "4K" {
		t.Fatalf("quality high imageSize = %q, want 4K", got)
	}
	if got := gjson.GetBytes(cfg, "candidateCount").Int(); got != 3 {
		t.Fatalf("candidateCount = %d, want 3", got)
	}
}

func TestGeminiImagesOptionsFromJSONPrefersExplicitGeminiSpelling(t *testing.T) {
	raw := []byte(`{"model":"gemini-3.1-flash-image","prompt":"p","size":"1024x1024","quality":"low","aspect_ratio":"4:3","image_size":"4K","n":2}`)
	opts := geminiImagesOptionsFromJSON(raw)
	if opts.AspectRatio != "4:3" || opts.ImageSize != "4K" || opts.Size != "1024x1024" || opts.Quality != "low" || opts.N != 2 {
		t.Fatalf("opts = %+v", opts)
	}

	raw = []byte(`{"image_config":{"aspect_ratio":"21:9","image_size":"1K"}}`)
	opts = geminiImagesOptionsFromJSON(raw)
	if opts.AspectRatio != "21:9" || opts.ImageSize != "1K" {
		t.Fatalf("image_config opts = %+v", opts)
	}
}

func TestBuildGeminiImagesPayload(t *testing.T) {
	payload, err := buildGeminiImagesPayload("draw a cat", []string{
		"data:image/png;base64,AA==",
	}, geminiImagesOptions{Size: "1536x1024"})
	if err != nil {
		t.Fatalf("buildGeminiImagesPayload: %v", err)
	}
	if got := gjson.GetBytes(payload, "contents.0.role").String(); got != "user" {
		t.Fatalf("role = %q, want user", got)
	}
	if got := gjson.GetBytes(payload, "contents.0.parts.0.text").String(); got != "draw a cat" {
		t.Fatalf("prompt part = %q", got)
	}
	if got := gjson.GetBytes(payload, "contents.0.parts.1.inlineData.mimeType").String(); got != "image/png" {
		t.Fatalf("inline mimeType = %q, want image/png", got)
	}
	if got := gjson.GetBytes(payload, "contents.0.parts.1.inlineData.data").String(); got != "AA==" {
		t.Fatalf("inline data = %q, want AA==", got)
	}
	if got := gjson.GetBytes(payload, "generationConfig.imageConfig.aspectRatio").String(); got != "3:2" {
		t.Fatalf("aspectRatio = %q, want 3:2", got)
	}

	if _, err := buildGeminiImagesPayload("p", []string{"https://example.com/cat.png"}, geminiImagesOptions{}); err == nil {
		t.Fatal("expected https image input to be rejected for Gemini image models")
	}
	if _, err := buildGeminiImagesPayload("p", []string{"data:image/png,AA=="}, geminiImagesOptions{}); err == nil {
		t.Fatal("expected non-base64 data URL to be rejected")
	}
}

const geminiImagesResponseFixture = `{
	"candidates": [{
		"content": {
			"role": "model",
			"parts": [
				{"text": "here is your cat"},
				{"inlineData": {"mimeType": "image/jpeg", "data": "aW1hZ2U="}}
			]
		},
		"finishReason": "STOP"
	}],
	"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 345, "totalTokenCount": 357}
}`

func TestBuildImagesAPIResponseFromGemini(t *testing.T) {
	out, err := buildImagesAPIResponseFromGemini([]byte(geminiImagesResponseFixture), "")
	if err != nil {
		t.Fatalf("buildImagesAPIResponseFromGemini: %v", err)
	}
	if got := gjson.GetBytes(out, "data.0.b64_json").String(); got != "aW1hZ2U=" {
		t.Fatalf("b64_json = %q", got)
	}
	if got := gjson.GetBytes(out, "data.0.mime_type").String(); got != "image/jpeg" {
		t.Fatalf("mime_type = %q, want image/jpeg", got)
	}
	if got := gjson.GetBytes(out, "usage.total_tokens").Int(); got != 357 {
		t.Fatalf("usage.total_tokens = %d, want 357", got)
	}
	if got := gjson.GetBytes(out, "usage.prompt_tokens").Int(); got != 12 {
		t.Fatalf("usage.prompt_tokens = %d, want 12", got)
	}
	if got := gjson.GetBytes(out, "usage.completion_tokens").Int(); got != 345 {
		t.Fatalf("usage.completion_tokens = %d, want 345", got)
	}
	if !gjson.GetBytes(out, "created").Exists() {
		t.Fatal("created timestamp missing")
	}

	out, err = buildImagesAPIResponseFromGemini([]byte(geminiImagesResponseFixture), "url")
	if err != nil {
		t.Fatalf("buildImagesAPIResponseFromGemini(url): %v", err)
	}
	if got := gjson.GetBytes(out, "data.0.url").String(); !strings.HasPrefix(got, "data:image/jpeg;base64,aW1hZ2U=") {
		t.Fatalf("url = %q, want data URL", got)
	}

	if _, err := buildImagesAPIResponseFromGemini([]byte(`{"candidates":[{"content":{"parts":[{"text":"no image"}]}}]}`), ""); err == nil {
		t.Fatal("expected error when upstream returns no image data")
	}
	if _, err := buildImagesAPIResponseFromGemini([]byte(`{"error":{"message":"boom"}}`), ""); err == nil {
		t.Fatal("expected error when upstream returns an error object")
	}
}

func TestExtractGeminiImagesResponseHandlesSnakeCaseMetadata(t *testing.T) {
	payload := []byte(`{"candidates":[{"content":{"parts":[{"inline_data":{"mime_type":"image/webp","data":"d2VicA=="}}]}}],"usage_metadata":{"prompt_token_count":5,"candidates_token_count":7,"total_token_count":12}}`)
	images, usage, err := extractGeminiImagesResponse(payload)
	if err != nil {
		t.Fatalf("extractGeminiImagesResponse: %v", err)
	}
	if len(images) != 1 || images[0].MimeType != "image/webp" || images[0].B64JSON != "d2VicA==" {
		t.Fatalf("images = %+v", images)
	}
	if got := gjson.GetBytes(usage, "total_tokens").Int(); got != 12 {
		t.Fatalf("total_tokens = %d, want 12", got)
	}
}

func TestImagesGenerationsGeminiBranchRejectsPromptlessRequest(t *testing.T) {
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, nil)
	handler := NewOpenAIAPIHandler(base)
	body := strings.NewReader(`{"model":"gemini-3.1-flash-image"}`)

	resp := performImagesEndpointRequest(t, imagesGenerationsPath, "application/json", body, handler.ImagesGenerations)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusBadRequest, resp.Body.String())
	}
}

func TestImagesEditsJSONGeminiBranchRequiresImage(t *testing.T) {
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, nil)
	handler := NewOpenAIAPIHandler(base)
	body := strings.NewReader(`{"model":"gemini-3.1-flash-image","prompt":"add a hat"}`)

	resp := performImagesEndpointRequest(t, imagesEditsPath, "application/json", body, handler.ImagesEdits)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusBadRequest, resp.Body.String())
	}
	if !strings.Contains(resp.Body.String(), "image is required") {
		t.Fatalf("body = %s, want image-required error", resp.Body.String())
	}
}

func TestImagesEditsJSONGeminiBranchRejectsRemoteImageURL(t *testing.T) {
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, nil)
	handler := NewOpenAIAPIHandler(base)
	body := strings.NewReader(`{"model":"gemini-3.1-flash-image","prompt":"add a hat","images":[{"image_url":"https://example.com/cat.png"}]}`)

	resp := performImagesEndpointRequest(t, imagesEditsPath, "application/json", body, handler.ImagesEdits)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusBadRequest, resp.Body.String())
	}
	if !strings.Contains(resp.Body.String(), "data URL") {
		t.Fatalf("body = %s, want data-URL error", resp.Body.String())
	}
}
