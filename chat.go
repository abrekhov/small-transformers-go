package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"strings"

	"small-transformers-go/internal/gpt"
	"small-transformers-go/internal/tok"
)

const chatEnd = "\nEND\n"

type chatExample struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

type chatPair struct {
	ids         []int
	prefix, ctx int
}

func chatPrefix(question string) string {
	return "\nUSER:\n" + strings.Join(strings.Fields(question), " ") + "\nBARD:\n"
}

func readChat(path string) ([]chatExample, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var examples []chatExample
	for {
		var example chatExample
		if err := dec.Decode(&example); err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("%s example %d: %w", path, len(examples)+1, err)
		}
		example.Question = strings.Join(strings.Fields(example.Question), " ")
		example.Answer = strings.TrimSpace(example.Answer)
		if example.Question == "" || example.Answer == "" || strings.Contains(example.Answer, chatEnd) {
			return nil, fmt.Errorf("%s example %d: question and answer must be nonempty; END is reserved", path, len(examples)+1)
		}
		examples = append(examples, example)
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("empty chat dataset: %s", path)
	}
	return examples, nil
}

func prepareChat(examples []chatExample, tokenizer *tok.Tokenizer, ctx int) ([]chatPair, error) {
	pairs := make([]chatPair, len(examples))
	for i, example := range examples {
		prefix := chatPrefix(example.Question)
		ids, err := tokenizer.Encode([]rune(prefix + example.Answer + chatEnd))
		if err != nil {
			return nil, err
		}
		if len(ids)-1 > ctx {
			return nil, fmt.Errorf("chat example %d needs %d context characters, model has %d", i+1, len(ids)-1, ctx)
		}
		prefixLen := len([]rune(prefix))
		pairs[i] = chatPair{ids: ids, prefix: prefixLen, ctx: ctx}
	}
	return pairs, nil
}

func chatBatch(pairs []chatPair, batch int, rng *rand.Rand, augment bool, focus int) ([]int, []int, int) {
	ctx := pairs[0].ctx
	idx, targets := make([]int, batch*ctx), make([]int, batch*ctx)
	for i := range targets {
		targets[i] = -1
	}
	weight := 0
	for i := 0; i < batch; i++ {
		j, offset := i, 0
		if rng != nil {
			j = rng.IntN(len(pairs))
		}
		pair := pairs[j]
		used := len(pair.ids) - 1
		if rng != nil && augment && rng.Float64() < .5 {
			offset = rng.IntN(ctx - used + 1)
		}
		start := i*ctx + offset
		copy(idx[start:start+used], pair.ids[:used])
		end := used
		if focus > 0 {
			end = min(end, pair.prefix-1+focus)
		}
		for k := pair.prefix - 1; k < end; k++ {
			targets[start+k] = pair.ids[k+1]
		}
		weight += end - pair.prefix + 1
	}
	return idx, targets, weight
}

func chatLoss(m *gpt.Model, pairs []chatPair, batch, batches, focus int) (float64, error) {
	rng := rand.New(rand.NewPCG(2025, 2026))
	total, weight := 0.0, 0
	for i := 0; batches > 0 && i < batches || batches == 0 && i < len(pairs); {
		var idx, targets []int
		var count int
		if batches == 0 {
			countBatch := min(batch, len(pairs)-i)
			idx, targets, count = chatBatch(pairs[i:], countBatch, nil, false, focus)
			i += countBatch
		} else {
			idx, targets, count = chatBatch(pairs, batch, rng, false, focus)
			i++
		}
		loss := float64(m.Loss(idx, targets))
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return 0, fmt.Errorf("non-finite chat validation loss")
		}
		total += loss * float64(count)
		weight += count
	}
	return total / float64(weight), nil
}

func chat(args []string) error {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	model := fs.String("model", "checkpoints/shakespeare-chat.bin", "chat-trained checkpoint")
	prompt := fs.String("prompt", "", "single message; omit for interactive input")
	n := fs.Int("n", 200, "maximum generated characters")
	temp := fs.Float64("temp", .7, "positive sampling temperature")
	topk := fs.Int("topk", 20, "top-k sampling; 0 disables filtering")
	seed := fs.Uint64("seed", 7, "sampling seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *n < 1 || *temp <= 0 || math.IsNaN(*temp) || math.IsInf(*temp, 0) || float32(*temp) == 0 || math.IsInf(float64(float32(*temp)), 0) || *topk < 0 {
		return fmt.Errorf("n and temp must be positive and finite, topk nonnegative; no positional arguments allowed")
	}
	m, vocab, err := gpt.Load(*model)
	if err != nil {
		return err
	}
	if !m.Cfg.Chat {
		return fmt.Errorf("checkpoint is not chat-trained; use train -chat -validation <jsonl>")
	}
	tokenizer := tok.New(vocab)
	stop, err := tokenizer.Encode([]rune(chatEnd))
	if err != nil {
		return err
	}
	rng := rand.New(rand.NewPCG(*seed, *seed))
	answer := func(question string) error {
		if strings.TrimSpace(question) == "" {
			return fmt.Errorf("message must be nonempty")
		}
		ids, err := tokenizer.Encode([]rune(chatPrefix(question)))
		if err != nil {
			return err
		}
		if len(ids) >= m.Cfg.Ctx {
			return fmt.Errorf("message is too long for the %d-character context", m.Cfg.Ctx)
		}
		out := m.GenerateUntil(ids, *n, float32(*temp), *topk, rng, stop)
		reply := strings.TrimSpace(string(tokenizer.Decode(out[len(ids):])))
		if reply == "" {
			return fmt.Errorf("model generated an empty reply")
		}
		fmt.Println(reply)
		return nil
	}
	if *prompt != "" {
		return answer(*prompt)
	}
	info, err := os.Stdin.Stat()
	if err != nil {
		return err
	}
	interactive := info.Mode()&os.ModeCharDevice != 0
	if interactive {
		fmt.Fprintln(os.Stderr, "Shakespeare-style chat. /quit exits. Each message is answered independently.")
	}
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	for {
		if interactive {
			fmt.Print("You> ")
		}
		if !scanner.Scan() {
			break
		}
		question := strings.TrimSpace(scanner.Text())
		if question == "/quit" || question == "/exit" {
			break
		}
		if question == "" {
			continue
		}
		if err := answer(question); err != nil {
			if !interactive {
				return err
			}
			fmt.Fprintln(os.Stderr, err)
		}
	}
	return scanner.Err()
}
