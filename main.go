package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"small-transformers-go/internal/gpt"
	"small-transformers-go/internal/mat"
	"small-transformers-go/internal/tok"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: small-transformers-go <download|train|eval|generate|chat> [flags]")
	}
	switch args[0] {
	case "download":
		return download(args[1:])
	case "train":
		return train(args[1:])
	case "eval":
		return evaluate(args[1:])
	case "chat":
		return chat(args[1:])
	case "generate":
		return generate(args[1:])
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func download(args []string) error {
	fs := flag.NewFlagSet("download", flag.ContinueOnError)
	url := fs.String("url", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "dataset URL")
	out := fs.String("out", "data/shakespeare.txt", "output path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected arguments: %v", fs.Args())
	}
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Get(*url)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
	}
	dir := filepath.Dir(*out)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".download-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	n, err := io.Copy(f, resp.Body)
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("empty dataset")
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(f.Name(), *out); err != nil {
		return err
	}
	fmt.Printf("downloaded %d bytes to %s\n", n, *out)
	return nil
}

func train(args []string) error {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	data := fs.String("data", "data/shakespeare.txt", "corpus path")
	out := fs.String("out", "model.bin", "checkpoint path")
	init := fs.String("init", "", "warm-start weights and architecture from checkpoint; reset optimizer")
	steps := fs.Int("steps", 2000, "training steps")
	batch := fs.Int("batch", 16, "sequences per batch")
	seq := fs.Int("seq", 64, "context length")
	dim := fs.Int("dim", 128, "embedding dimension")
	heads := fs.Int("heads", 4, "attention heads")
	layers := fs.Int("layers", 4, "transformer blocks")
	lr := fs.Float64("lr", 0.001, "peak learning rate")
	seed := fs.Uint64("seed", 42, "random seed")
	evalEvery := fs.Int("eval-every", 200, "validation and checkpoint interval")
	evalBatches := fs.Int("eval-batches", 8, "fixed validation batches")
	split := fs.Float64("split", 0.9, "training fraction; 1 disables validation")
	chatData := fs.Bool("chat", false, "supervised question/answer JSONL; train on answers only")
	validationFile := fs.String("validation", "", "separate held-out JSONL for chat training")
	chatFocus := fs.Int("chat-focus", 32, "chat validation prefix length; half updates focus here; 0 uses full replies")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *chatFocus < 0 || *steps < 1 || *batch < 1 || *evalEvery < 1 || *evalBatches < 1 || *lr <= 0 || math.IsNaN(*lr) || math.IsInf(*lr, 0) || float32(*lr) == 0 || math.IsInf(float64(float32(*lr)), 0) {
		return fmt.Errorf("steps, batch, eval-every, eval-batches and lr must be positive; no positional arguments allowed")
	}
	cfg := gpt.Config{Vocab: 1, Dim: *dim, Heads: *heads, Layers: *layers, Ctx: *seq}
	var m *gpt.Model
	var initVocab []rune
	if *init != "" {
		loaded, vocab, err := gpt.Load(*init)
		if err != nil {
			return err
		}
		m, initVocab, cfg = loaded, vocab, loaded.Cfg
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	var err error
	var tkz *tok.Tokenizer
	var trainTokens, valTokens []int
	var trainPairs, valPairs []chatPair
	if *chatData {
		m, tkz, trainPairs, valPairs, err = setupChat(m, cfg, initVocab, *data, *validationFile, *seed)
		if err != nil {
			return err
		}
		cfg = m.Cfg
		fmt.Printf("chat validation focuses on the first %d reply characters; 0 means full reply\n", *chatFocus)
		fmt.Printf("chat examples: %d train, %d validation; answer-only loss; vocab: %d; backend: %s\n", len(trainPairs), len(valPairs), len(tkz.Vocab), mat.Backend)
	} else {
		raw, err := os.ReadFile(*data)
		if err != nil {
			return err
		}
		runes := []rune(string(raw))
		if len(runes) <= cfg.Ctx {
			return fmt.Errorf("corpus must contain at least seq+1 characters")
		}
		tkz = tok.Build(runes)
		if m != nil {
			tkz = tok.New(initVocab)
		}
		tokens, err := tkz.Encode(runes)
		if err != nil {
			return err
		}
		trainTokens, valTokens, err = splitTokens(tokens, *split, cfg.Ctx)
		if err != nil {
			return err
		}
		if m == nil {
			cfg.Vocab = len(tkz.Vocab)
			m = gpt.New(cfg, *seed)
		}
		fmt.Printf("corpus: %d chars, vocab: %d, backend: %s\n", len(runes), len(tkz.Vocab), mat.Backend)
		fmt.Printf("train chars: %d, validation chars: %d\n", len(trainTokens), len(valTokens))
	}
	if *init != "" {
		fmt.Printf("warm start from %s; optimizer reset, architecture from checkpoint\n", *init)
	}
	fmt.Println(m.Describe())
	hasValidation := len(valTokens) > 0 || len(valPairs) > 0
	validate := func() (float64, error) {
		if *chatData {
			return chatLoss(m, valPairs, *batch, *evalBatches, *chatFocus)
		}
		return validationLoss(m, valTokens, *batch, *evalBatches)
	}
	adam := gpt.NewAdam(m)
	rng := rand.New(rand.NewPCG(*seed, *seed+1))
	warmup := min(100, *steps/10)
	start := time.Now()
	bestLoss, bestStep := math.Inf(1), 0
	if *init != "" && hasValidation {
		bestLoss, err = validate()
		if err != nil {
			return err
		}
		if err := m.Save(*out, tkz.Vocab); err != nil {
			return err
		}
		fmt.Printf("initial validation %.4f\n", bestLoss)
	}
	for step := 1; step <= *steps; step++ {
		lrNow := schedule(*lr, step, *steps, warmup)
		var idx, tgt []int
		if *chatData {
			focus := 0
			if *chatFocus > 0 && rng.Float64() < .5 {
				focus = *chatFocus
			}
			idx, tgt, _ = chatBatch(trainPairs, *batch, rng, true, focus)
		} else {
			idx, tgt = sampleBatch(trainTokens, *batch, cfg.Ctx, rng)
		}
		loss, err := m.TrainStep(idx, tgt, adam, lrNow)
		if err != nil {
			return fmt.Errorf("step %d: %w", step, err)
		}
		if step%*evalEvery == 0 || step == 1 || step == *steps {
			elapsed := time.Since(start)
			tokensPerSecond := float64(step) * float64(*batch) * float64(cfg.Ctx) / elapsed.Seconds()
			validation := ""
			if hasValidation {
				val, err := validate()
				if err != nil {
					return err
				}
				validation = fmt.Sprintf(" val %.4f ppl %.2f", val, math.Exp(val))
				if val < bestLoss {
					if err := m.Save(*out, tkz.Vocab); err != nil {
						return err
					}
					bestLoss, bestStep = val, step
				}
			}
			fmt.Printf("step %d/%d loss %.4f%s lr %.6f tokens/s %.0f elapsed %s\n", step, *steps, loss, validation, lrNow, tokensPerSecond, elapsed.Round(time.Millisecond))
		}
	}
	if !hasValidation {
		if err := m.Save(*out, tkz.Vocab); err != nil {
			return err
		}
	} else {
		fmt.Printf("best validation %.4f at step %d\n", bestLoss, bestStep)
	}
	fmt.Printf("saved checkpoint to %s\n", *out)
	return nil
}

func schedule(base float64, step, steps, warmup int) float32 {
	if step <= warmup {
		return float32(base * float64(step) / float64(warmup))
	}
	p := float64(step-warmup) / math.Max(1, float64(steps-warmup))
	return float32(base * (0.1 + 0.9*0.5*(1+math.Cos(math.Pi*p))))
}

func sampleBatch(tokens []int, batch, seq int, rng *rand.Rand) (idx, tgt []int) {
	idx = make([]int, batch*seq)
	tgt = make([]int, batch*seq)
	for b := 0; b < batch; b++ {
		pos := rng.IntN(len(tokens) - seq)
		copy(idx[b*seq:(b+1)*seq], tokens[pos:pos+seq])
		copy(tgt[b*seq:(b+1)*seq], tokens[pos+1:pos+seq+1])
	}
	return idx, tgt
}

func splitTokens(tokens []int, split float64, seq int) ([]int, []int, error) {
	if split <= 0 || split > 1 || math.IsNaN(split) {
		return nil, nil, fmt.Errorf("split must be in (0, 1]")
	}
	cut := int(float64(len(tokens)) * split)
	if cut <= seq || split < 1 && len(tokens)-cut <= seq {
		return nil, nil, fmt.Errorf("each split must contain at least seq+1 characters; use -split 1 to disable validation")
	}
	return tokens[:cut], tokens[cut:], nil
}

func validationLoss(m *gpt.Model, tokens []int, batch, batches int) (float64, error) {
	if batches == 0 {
		var total float64
		for start := 0; start < len(tokens)-1; start += m.Cfg.Ctx {
			end := min(start+m.Cfg.Ctx, len(tokens)-1)
			loss := float64(m.Loss(tokens[start:end], tokens[start+1:end+1]))
			if math.IsNaN(loss) || math.IsInf(loss, 0) {
				return 0, fmt.Errorf("non-finite validation loss")
			}
			total += loss * float64(end-start)
		}
		return total / float64(len(tokens)-1), nil
	}
	rng := rand.New(rand.NewPCG(2025, 2026))
	var total float64
	for i := 0; i < batches; i++ {
		idx, tgt := sampleBatch(tokens, batch, m.Cfg.Ctx, rng)
		loss := float64(m.Loss(idx, tgt))
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return 0, fmt.Errorf("non-finite validation loss")
		}
		total += loss
	}
	return total / float64(batches), nil
}

func evaluate(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	model := fs.String("model", "model.bin", "checkpoint path")
	data := fs.String("data", "data/shakespeare.txt", "original corpus path")
	split := fs.Float64("split", 0.9, "training fraction; evaluate the remaining tail")
	batch := fs.Int("batch", 16, "sequences per evaluation batch")
	batches := fs.Int("batches", 8, "fixed validation batches; 0 scans the complete tail")
	chatData := fs.Bool("chat", false, "evaluate answer-only loss on held-out question/answer JSONL")
	chatFocus := fs.Int("chat-focus", 0, "score the first N reply characters; 0 scores full replies")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *chatFocus < 0 || *batch < 1 || *batches < 0 || (!*chatData && *split == 1) {
		return fmt.Errorf("batch must be positive, batches nonnegative; evaluation requires a held-out split")
	}
	m, vocab, err := gpt.Load(*model)
	if err != nil {
		return err
	}
	if *chatData {
		examples, err := readChat(*data)
		if err != nil {
			return err
		}
		pairs, err := prepareChat(examples, tok.New(vocab), m.Cfg.Ctx)
		if err != nil {
			return err
		}
		loss, err := chatLoss(m, pairs, *batch, *batches, *chatFocus)
		if err != nil {
			return err
		}
		fmt.Printf("chat validation loss %.4f perplexity %.2f on %d examples (answer characters only; prefix %d, 0 means full reply)\n", loss, math.Exp(loss), len(examples), *chatFocus)
		return nil
	}
	raw, err := os.ReadFile(*data)
	if err != nil {
		return err
	}
	tokens, err := tok.New(vocab).Encode([]rune(string(raw)))
	if err != nil {
		return err
	}
	_, valTokens, err := splitTokens(tokens, *split, m.Cfg.Ctx)
	if err != nil {
		return err
	}
	loss, err := validationLoss(m, valTokens, *batch, *batches)
	if err != nil {
		return err
	}
	predictions, mode := *batch**batches*m.Cfg.Ctx, "sampled"
	if *batches == 0 {
		predictions, mode = len(valTokens)-1, "full scan"
	}
	fmt.Printf("validation loss %.4f perplexity %.2f on %d held-out chars (%d predictions; %s)\n", loss, math.Exp(loss), len(valTokens), predictions, mode)
	return nil
}

func generate(args []string) error {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	model := fs.String("model", "model.bin", "checkpoint path")
	prompt := fs.String("prompt", "\n", "initial text")
	n := fs.Int("n", 300, "new characters")
	temp := fs.Float64("temp", 0.8, "positive sampling temperature")
	topk := fs.Int("topk", 40, "top-k sampling; 0 disables filtering")
	seed := fs.Uint64("seed", 7, "sampling seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *prompt == "" || *n < 0 || *temp <= 0 || math.IsNaN(*temp) || math.IsInf(*temp, 0) || float32(*temp) == 0 || math.IsInf(float64(float32(*temp)), 0) || *topk < 0 {
		return fmt.Errorf("prompt must be nonempty, n and topk nonnegative, temp positive and finite; no positional arguments allowed")
	}
	m, vocab, err := gpt.Load(*model)
	if err != nil {
		return err
	}
	tkz := tok.New(vocab)
	ids, err := tkz.Encode([]rune(*prompt))
	if err != nil {
		return err
	}
	start := time.Now()
	out := m.Generate(ids, *n, float32(*temp), *topk, rand.New(rand.NewPCG(*seed, *seed)))
	fmt.Print(string(tkz.Decode(out)))
	fmt.Fprintf(os.Stderr, "\n[%d tokens in %s]\n", *n, time.Since(start).Round(time.Millisecond))
	return nil
}
