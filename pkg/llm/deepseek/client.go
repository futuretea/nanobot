package deepseek

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/nanobot-ai/nanobot/pkg/complete"
	"github.com/nanobot-ai/nanobot/pkg/llm/progress"
	"github.com/nanobot-ai/nanobot/pkg/log"
	"github.com/nanobot-ai/nanobot/pkg/mcp"
	"github.com/nanobot-ai/nanobot/pkg/types"
)

type Client struct {
	Config
}

type Config struct {
	APIKey  string
	BaseURL string
	Headers map[string]string
}

func NewClient(cfg Config) *Client {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.deepseek.com/v1"
	}
	if cfg.Headers == nil {
		cfg.Headers = map[string]string{}
	}
	if _, ok := cfg.Headers["Authorization"]; !ok && cfg.APIKey != "" {
		cfg.Headers["Authorization"] = "Bearer " + cfg.APIKey
	}
	if _, ok := cfg.Headers["Content-Type"]; !ok {
		cfg.Headers["Content-Type"] = "application/json"
	}

	return &Client{
		Config: cfg,
	}
}

func (c *Client) Complete(ctx context.Context, completionRequest types.CompletionRequest, opts ...types.CompletionOptions) (*types.CompletionResponse, error) {
	req, err := toRequest(&completionRequest)
	if err != nil {
		return nil, err
	}

	ts := time.Now()
	resp, err := c.complete(ctx, completionRequest.Agent, req, opts...)
	if err != nil {
		return nil, err
	}

	return toResponse(resp, ts)
}

func (c *Client) complete(ctx context.Context, agentName string, req Request, opts ...types.CompletionOptions) (*Response, error) {
	opt := complete.Complete(opts...)
	req.Stream = true

	data, _ := json.Marshal(req)
	log.Messages(ctx, "deepseek-api", true, data)
	
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	
	for key, value := range c.Headers {
		httpReq.Header.Set(key, value)
	}

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()
	
	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, fmt.Errorf("failed to get response from DeepSeek API: %s %q", httpResp.Status, string(body))
	}

	var (
		lines = bufio.NewScanner(httpResp.Body)
		resp  Response
	)

	for lines.Scan() {
		line := lines.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		data = strings.TrimSpace(data)
		
		if data == "[DONE]" {
			break
		}

		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Errorf(ctx, "failed to decode chunk: %v: %s", err, data)
			continue
		}

		if resp.ID == "" {
			resp.ID = chunk.ID
			resp.Model = chunk.Model
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		
		// Handle content delta
		if choice.Delta.Content != "" {
			resp.Choices = []Choice{{
				Message: Message{
					Role:    "assistant",
					Content: resp.GetContent() + choice.Delta.Content,
				},
			}}
			
				progress.Send(ctx, &types.CompletionProgress{
					Model:     resp.Model,
					Agent:     agentName,
					MessageID: resp.ID,
					Item: types.CompletionItem{
						ID:      fmt.Sprintf("%s-0", resp.ID),
						Partial: true,
						HasMore: true,
						Content: &mcp.Content{
							Type: "text",
							Text: choice.Delta.Content,
						},
					},
				}, opt.ProgressToken)
		}

		// Handle tool calls
		if len(choice.Delta.ToolCalls) > 0 {
			if len(resp.Choices) == 0 {
				resp.Choices = []Choice{{
					Message: Message{
						Role: "assistant",
					},
				}}
			}
			
			for _, tc := range choice.Delta.ToolCalls {
				if tc.Index >= len(resp.Choices[0].Message.ToolCalls) {
					resp.Choices[0].Message.ToolCalls = append(resp.Choices[0].Message.ToolCalls, ToolCall{
						ID:   tc.ID,
						Type: tc.Type,
						Function: FunctionCall{
							Name:      tc.Function.Name,
							Arguments: tc.Function.Arguments,
						},
					})
				} else {
					resp.Choices[0].Message.ToolCalls[tc.Index].Function.Arguments += tc.Function.Arguments
				}
				
				progress.Send(ctx, &types.CompletionProgress{
					Model:     resp.Model,
					Agent:     agentName,
					MessageID: resp.ID,
					Item: types.CompletionItem{
						ID:      fmt.Sprintf("%s-%d", resp.ID, tc.Index),
						Partial: true,
						HasMore: true,
						ToolCall: &types.ToolCall{
							CallID:    tc.ID,
							Name:      tc.Function.Name,
							Arguments: tc.Function.Arguments,
						},
					},
				}, opt.ProgressToken)
			}
		}

		if choice.FinishReason != "" {
			resp.Choices[0].FinishReason = choice.FinishReason
		}
	}

	if err := lines.Err(); err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	respData, err := json.Marshal(resp)
	if err == nil {
		log.Messages(ctx, "deepseek-api", false, respData)
	}

	return &resp, nil
}
