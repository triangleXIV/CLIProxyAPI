package auth

import (
	"encoding/json"
	"strings"
)

type completionResponse struct {
	Object string            `json:"object"`
	Status string            `json:"status"`
	Error  json.RawMessage   `json:"error"`
	Output []json.RawMessage `json:"output"`
	Usage  *struct {
		OutputTokens tokenCount `json:"output_tokens"`
	} `json:"usage"`
}

// evalResponses recognizes only known Responses envelopes. Unknown events and
// output item types remain pass-through, rather than risking a false retry.
func (a *emptyCompletionAccum) evalResponses(data []byte) bool {
	var event struct {
		Type     string              `json:"type"`
		Object   string              `json:"object"`
		Response *completionResponse `json:"response"`
		Item     json.RawMessage     `json:"item"`
		Part     json.RawMessage     `json:"part"`
		Delta    json.RawMessage     `json:"delta"`
		Text     string              `json:"text"`
	}
	if json.Unmarshal(data, &event) != nil || (event.Object != "response" && !strings.HasPrefix(event.Type, "response.")) {
		return false
	}
	a.recognized = true
	a.sawMessageData = true
	if event.Object == "response" {
		event.Response = &completionResponse{}
		if json.Unmarshal(data, event.Response) != nil {
			a.sawUnknownData = true
			return true
		}
	} else {
		switch event.Type {
		case "response.created", "response.in_progress", "response.completed", "response.incomplete", "response.failed":
			if event.Response == nil {
				a.sawUnknownData = true
			}
		case "response.output_item.added", "response.output_item.done", "response.content_part.added", "response.content_part.done", "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
			if len(event.Item) == 0 && len(event.Part) == 0 {
				a.sawUnknownData = true
			}
		case "response.output_text.delta", "response.output_text.done", "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done", "response.reasoning_text.delta", "response.reasoning_text.done", "response.refusal.delta", "response.refusal.done", "response.function_call_arguments.delta", "response.function_call_arguments.done", "response.custom_tool_call_input.delta", "response.custom_tool_call_input.done":
			var delta string
			if len(event.Delta) > 0 && json.Unmarshal(event.Delta, &delta) != nil {
				a.sawUnknownData = true
			}
			if strings.TrimSpace(delta+event.Text) != "" {
				a.hasContent = true
			}
		default:
			a.sawUnknownData = true
		}
	}
	if event.Response != nil {
		response := event.Response
		if response.Object != "response" && response.Status == "" {
			a.sawUnknownData = true
		}
		if response.Usage != nil {
			a.sawUsage = true
			a.addUsage(response.Usage.OutputTokens)
		}
		if response.Status == "completed" || event.Type == "response.completed" {
			a.terminal = true
		}
		if response.Status == "incomplete" || response.Status == "failed" || event.Type == "response.incomplete" || event.Type == "response.failed" || nonEmptyJSONPayload(response.Error) {
			// Preserve failures and token-limit stops for their normal handlers.
			a.blocked = true
		}
		for _, item := range response.Output {
			a.evalResponseItem(item)
		}
	}
	if len(event.Item) > 0 {
		a.evalResponseItem(event.Item)
	}
	if len(event.Part) > 0 {
		a.evalResponseItem(event.Part)
	}
	return true
}

func (a *emptyCompletionAccum) evalResponseItem(raw json.RawMessage) {
	var item struct {
		Type             string            `json:"type"`
		Text             string            `json:"text"`
		Refusal          string            `json:"refusal"`
		EncryptedContent string            `json:"encrypted_content"`
		Content          []json.RawMessage `json:"content"`
		Summary          []json.RawMessage `json:"summary"`
		Result           json.RawMessage   `json:"result"`
	}
	if json.Unmarshal(raw, &item) != nil {
		a.sawUnknownData = true
		return
	}
	switch item.Type {
	case "message", "reasoning", "output_text", "summary_text", "reasoning_text", "refusal":
		if strings.TrimSpace(item.Text+item.Refusal) != "" {
			a.hasContent = true
		}
		if encrypted := strings.TrimSpace(item.EncryptedContent); encrypted != "" && !strings.HasPrefix(encrypted, "cpa-gemini-responses-carrier-v1:") {
			// Opaque provider reasoning may contain real encrypted output. A
			// Gemini replay carrier alone, however, is only a signature.
			a.hasContent = true
		}
		for _, part := range append(item.Content, item.Summary...) {
			a.evalResponseItem(part)
		}
	case "function_call", "custom_tool_call":
		if isMeaningfulToolCall(raw) {
			a.hasToolCalls = true
		}
	case "image_generation_call":
		if nonEmptyJSONPayload(item.Result) {
			a.hasContent = true
		}
	default:
		a.sawUnknownData = true
	}
}

func (a *emptyCompletionAccum) observe(payload []byte) {
	if isSSEPayload(payload) {
		a.evalSSE(payload)
	} else if !a.evalJSON(payload) {
		a.sawUnknownData = true
	}
}

func (a *emptyCompletionAccum) ready() bool {
	return a.sawUnknownData || a.blocked || a.hasContent || a.hasToolCalls || a.completionTokens > 0
}
