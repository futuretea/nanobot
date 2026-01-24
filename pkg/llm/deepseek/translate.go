package deepseek

import (
	"encoding/base64"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/nanobot-ai/nanobot/pkg/mcp"
	"github.com/nanobot-ai/nanobot/pkg/types"
)

func toResponse(resp *Response, created time.Time) (*types.CompletionResponse, error) {
	result := &types.CompletionResponse{
		Model: resp.Model,
		Output: types.Message{
			ID:      resp.ID,
			Created: &created,
			Role:    "assistant",
		},
	}

	if len(resp.Choices) == 0 {
		return result, nil
	}

	choice := resp.Choices[0]
	
	// Handle text content
	if choice.Message.Content != "" {
		result.Output.Items = append(result.Output.Items, types.CompletionItem{
			ID: fmt.Sprintf("%s-0", resp.ID),
			Content: &mcp.Content{
				Type: "text",
				Text: choice.Message.Content,
			},
		})
	}

	// Handle tool calls
	for i, tc := range choice.Message.ToolCalls {
		result.Output.Items = append(result.Output.Items, types.CompletionItem{
			ID: fmt.Sprintf("%s-%d", resp.ID, i),
			ToolCall: &types.ToolCall{
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		})
	}

	return result, nil
}

func toRequest(req *types.CompletionRequest) (Request, error) {
	if req.MaxTokens == 0 {
		req.MaxTokens = 4096
	}

	var temp float64
	if req.Temperature != nil {
		temp, _ = req.Temperature.Float64()
	}

	var topP float64
	if req.TopP != nil {
		topP, _ = req.TopP.Float64()
	}

	result := Request{
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: temp,
		TopP:        topP,
	}

	// Convert tools
	for _, tool := range req.Tools {
		result.Tools = append(result.Tools, Tool{
			Type: "function",
			Function: Function{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
			},
		})
	}

	// Handle tool choice
	if req.ToolChoice != "" {
		switch req.ToolChoice {
		case "auto":
			result.ToolChoice = "auto"
		case "none":
			result.ToolChoice = "none"
		case "required":
			result.ToolChoice = "required"
		default:
			// Specific tool name
			result.ToolChoice = map[string]any{
				"type": "function",
				"function": map[string]string{
					"name": req.ToolChoice,
				},
			}
		}
	}

	// Add system prompt as first message if present
	if strings.TrimSpace(req.SystemPrompt) != "" {
		result.Messages = append(result.Messages, Message{
			Role:    "system",
			Content: strings.TrimSpace(req.SystemPrompt),
		})
	}

	// Convert messages
	for _, msg := range req.Input {
		for _, input := range msg.Items {
			if input.Content != nil {
				content := convertContent(*input.Content)
				if content != "" {
					result.Messages = append(result.Messages, Message{
						Role:    msg.Role,
						Content: content,
					})
				}
			}
			
			if input.ToolCall != nil {
				result.Messages = append(result.Messages, Message{
					Role: "assistant",
					ToolCalls: []ToolCall{
						{
							ID:   input.ToolCall.CallID,
							Type: "function",
							Function: FunctionCall{
								Name:      input.ToolCall.Name,
								Arguments: input.ToolCall.Arguments,
							},
						},
					},
				})
			}
			
			if input.ToolCallResult != nil {
				var resultContent string
				for _, content := range input.ToolCallResult.Output.Content {
					if content.Type == "text" {
						if resultContent != "" {
							resultContent += "\n"
						}
						resultContent += content.Text
					}
				}
				
				result.Messages = append(result.Messages, Message{
					Role:       "tool",
					Content:    resultContent,
					ToolCallID: input.ToolCallResult.CallID,
				})
			}
		}
	}

	return result, nil
}

func convertContent(content mcp.Content) string {
	if content.Type == "text" || content.Type == "" {
		return content.Text
	}
	
	if content.Type == "resource" && content.Resource != nil && 
		content.Resource.Annotations != nil && 
		slices.Contains(content.Resource.Annotations.Audience, "assistant") {
		
		if _, ok := types.TextMimeTypes[content.Resource.MIMEType]; ok {
			if content.Resource.Blob != "" {
				text, _ := base64.StdEncoding.DecodeString(content.Resource.Blob)
				return string(text)
			} else if content.Resource.Text != "" {
				return content.Resource.Text
			}
		}
		
		// For non-text resources, return a reference
		return fmt.Sprintf("[Resource: %s]", content.Resource.URI)
	}
	
	if content.Type == "image" {
		return "[Image content not supported in text format]"
	}
	
	return ""
}
