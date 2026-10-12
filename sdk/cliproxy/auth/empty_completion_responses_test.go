package auth

import (
	"context"
	"errors"
	"strings"
	"testing"

	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
)

func completionChunks(payloads ...string) <-chan cliproxyexecutor.StreamChunk {
	ch := make(chan cliproxyexecutor.StreamChunk, len(payloads))
	for _, payload := range payloads {
		ch <- cliproxyexecutor.StreamChunk{Payload: []byte(payload)}
	}
	close(ch)
	return ch
}

func TestReadStreamBootstrapRejectsSemanticEmptyCompletion(t *testing.T) {
	tests := []struct {
		name     string
		payloads []string
	}{
		{"responses", []string{`event: response.created
data: {"type":"response.created","response":{"object":"response","status":"in_progress","output":[]}}

`, `event: response.completed
data: {"type":"response.completed","response":{"object":"response","status":"completed","output":[],"usage":{"output_tokens":0}}}

`}},
		{"chat role then stop", []string{`{"choices":[{"delta":{"role":"assistant"}}]}`, `{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":0}}`}},
		{"header only EOF", []string{`{"type":"response.created","response":{"object":"response","status":"in_progress","output":[]}}`}},
		{"DONE only", []string{"data: [DONE]\n\n"}},
		{"Gemini signature only", []string{`{"type":"response.output_item.added","item":{"type":"reasoning","encrypted_content":"cpa-gemini-responses-carrier-v1:signature","summary":[]}}`, `{"type":"response.completed","response":{"object":"response","status":"completed","output":[{"type":"reasoning","encrypted_content":"cpa-gemini-responses-carrier-v1:signature","summary":[]}],"usage":{"output_tokens":0}}}`}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := readStreamBootstrap(context.Background(), completionChunks(tt.payloads...))
			var authErr *Error
			if !errors.As(err, &authErr) || authErr.Code != "empty_completion" {
				t.Fatalf("bootstrap error = %v, want empty_completion", err)
			}
		})
	}
}

func TestReadStreamBootstrapReleasesFirstContentBeforeEOF(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch := make(chan cliproxyexecutor.StreamChunk)
	prefix := `{"type":"response.created","response":{"object":"response","status":"in_progress","output":[]}}`
	content := `{"type":"response.output_text.delta","delta":"hello"}`
	go func() {
		defer close(ch)
		for _, payload := range []string{prefix, content} {
			select {
			case ch <- cliproxyexecutor.StreamChunk{Payload: []byte(payload)}:
			case <-ctx.Done():
				return
			}
		}
		// The upstream stays open until bootstrap returns and cancels it.
		<-ctx.Done()
	}()
	buffered, closed, err := readStreamBootstrap(ctx, ch)
	if err != nil || closed || len(buffered) != 2 || string(buffered[0].Payload) != prefix || string(buffered[1].Payload) != content {
		t.Fatalf("bootstrap = %v, closed=%v, err=%v", buffered, closed, err)
	}
}

func TestReadStreamBootstrapHonorsClientCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan cliproxyexecutor.StreamChunk, 1)
	ch <- cliproxyexecutor.StreamChunk{Payload: []byte(`{"choices":[{"delta":{"role":"assistant"}}]}`)}
	cancel()
	_, _, err := readStreamBootstrap(ctx, ch)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("bootstrap error = %v, want context.Canceled", err)
	}
}

func TestEmptyResponsesCompletionPayload(t *testing.T) {
	for _, payload := range []string{
		`{"object":"response","status":"completed","output":[],"usage":{"output_tokens":0}}`,
		`{"object":"response","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"  "}]}]}`,
	} {
		if !isEmptyCompletionPayload([]byte(payload)) {
			t.Fatalf("response should be empty: %s", payload)
		}
	}
}

func TestReadStreamBootstrapPreservesMeaningfulAndBlockedOutput(t *testing.T) {
	for name, payload := range map[string]string{
		"text":                       `{"type":"response.output_text.delta","delta":"hello"}`,
		"reasoning":                  `{"type":"response.reasoning_summary_text.delta","delta":"thinking"}`,
		"tool":                       `{"type":"response.output_item.added","item":{"type":"function_call","name":"lookup","arguments":""}}`,
		"image":                      `{"object":"response","status":"completed","output":[{"type":"image_generation_call","result":"image-bytes"}]}`,
		"positive usage":             `{"type":"response.completed","response":{"object":"response","status":"completed","output":[],"usage":{"output_tokens":2}}}`,
		"incomplete":                 `{"type":"response.incomplete","response":{"object":"response","status":"incomplete","output":[]}}`,
		"refusal":                    `{"type":"response.refusal.delta","delta":"cannot comply"}`,
		"content filter":             `{"choices":[{"delta":{},"finish_reason":"content_filter"}]}`,
		"length":                     `{"choices":[{"delta":{},"finish_reason":"length"}]}`,
		"opaque encrypted reasoning": `{"object":"response","status":"completed","output":[{"type":"reasoning","encrypted_content":"opaque-codex-value"}]}`,
		"unknown format":             `{"type":"new_provider_event","opaque":true}`,
		"native Gemini":              `{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			buffered, _, err := readStreamBootstrap(context.Background(), completionChunks(payload))
			if err != nil || len(buffered) != 1 || string(buffered[0].Payload) != payload {
				t.Fatalf("bootstrap = %v, %v; payload must pass through unchanged", buffered, err)
			}
			if isEmptyCompletionPayload([]byte(payload)) {
				t.Fatal("valid/blocked/unknown output must not rotate credentials")
			}
		})
	}
}

func TestManagerExecuteStreamRetriesEmptyResponsesBeforeDelivery(t *testing.T) {
	alias := "gemini-empty-response-regression"
	executor := &openAICompatPoolExecutor{
		id: openAICompatPoolProviderKey,
		streamPayloads: map[string][]cliproxyexecutor.StreamChunk{
			"empty-upstream": {
				{Payload: []byte(`{"type":"response.created","response":{"object":"response","status":"in_progress","output":[]}}`)},
				{Payload: []byte(`{"type":"response.completed","response":{"object":"response","status":"completed","output":[],"usage":{"output_tokens":0}}}`)},
			},
		},
	}
	manager := newOpenAICompatPoolTestManager(t, alias, []internalconfig.OpenAICompatibilityModel{
		{Name: "empty-upstream", Alias: alias},
		{Name: "healthy-upstream", Alias: alias},
	}, executor)
	result, err := manager.ExecuteStream(context.Background(), []string{openAICompatPoolProviderKey}, cliproxyexecutor.Request{Model: alias}, cliproxyexecutor.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var output strings.Builder
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatal(chunk.Err)
		}
		output.Write(chunk.Payload)
	}
	if output.String() != "healthy-upstream" {
		t.Fatalf("client received %q, want only the healthy attempt", output.String())
	}
	if got := executor.StreamModels(); len(got) != 2 || got[0] != "empty-upstream" || got[1] != "healthy-upstream" {
		t.Fatalf("attempts = %v", got)
	}
}
