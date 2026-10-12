package responses

import (
	"context"
	"fmt"
	"testing"

	"github.com/tidwall/gjson"
)

func TestGeminiResponsesContentFilterIsNotSuccessfulEmptyCompletion(t *testing.T) {
	for _, reason := range []string{"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "SPII", "BLOCKLIST", "IMAGE_SAFETY"} {
		t.Run(reason, func(t *testing.T) {
			raw := fmt.Sprintf(`{"candidates":[{"finishReason":%q}],"usageMetadata":{"candidatesTokenCount":0}}`, reason)
			out := ConvertGeminiResponseToOpenAIResponsesNonStream(context.Background(), "gemini-3.8-flash-high", nil, nil, []byte(raw), nil)
			if gjson.GetBytes(out, "status").String() != "incomplete" || gjson.GetBytes(out, "incomplete_details.reason").String() != "content_filter" {
				t.Fatalf("filtered non-stream output = %s", out)
			}
			var param any
			events := issue6258Translate(t, &param, raw)
			terminal := issue6258Terminal(t, events)
			if terminal.Get("status").String() != "incomplete" || terminal.Get("incomplete_details.reason").String() != "content_filter" {
				t.Fatalf("filtered stream output = %s", terminal.Raw)
			}
		})
	}
}
