package main

import (
	"fmt"
	"slices"
	"strings"

	"small-transformers-go/internal/gpt"
	"small-transformers-go/internal/tok"
)

func setupChat(m *gpt.Model, cfg gpt.Config, old []rune, data, validation string, seed uint64) (*gpt.Model, *tok.Tokenizer, []chatPair, []chatPair, error) {
	if validation == "" {
		return nil, nil, nil, nil, fmt.Errorf("chat training requires -validation with a separate JSONL dataset")
	}
	training, err := readChat(data)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	heldout, err := readChat(validation)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	seen := make(map[string]bool)
	for _, example := range training {
		seen[strings.ToLower(example.Question)] = true
	}
	for _, example := range heldout {
		if seen[strings.ToLower(example.Question)] {
			return nil, nil, nil, nil, fmt.Errorf("validation question also appears in training: %q", example.Question)
		}
	}
	runes := append([]rune{'\n'}, old...)
	for r := rune(32); r <= 126; r++ {
		runes = append(runes, r)
	}
	for _, examples := range [][]chatExample{training, heldout} {
		for _, example := range examples {
			runes = append(runes, []rune(chatPrefix(example.Question)+example.Answer+chatEnd)...)
		}
	}
	tokenizer := tok.Build(runes)
	trainPairs, err := prepareChat(training, tokenizer, cfg.Ctx)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	valPairs, err := prepareChat(heldout, tokenizer, cfg.Ctx)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	if m == nil {
		cfg.Vocab = len(tokenizer.Vocab)
		m = gpt.New(cfg, seed)
	} else if !slices.Equal(old, tokenizer.Vocab) {
		m = m.WithVocab(old, tokenizer.Vocab, seed)
	}
	m.Cfg.Chat = true
	return m, tokenizer, trainPairs, valPairs, nil
}
