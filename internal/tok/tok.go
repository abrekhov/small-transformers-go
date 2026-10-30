package tok

import (
	"fmt"
	"sort"
)

type Tokenizer struct {
	Vocab []rune
	ToID  map[rune]int
}

func Build(text []rune) *Tokenizer {
	set := map[rune]struct{}{}
	for _, r := range text {
		set[r] = struct{}{}
	}
	vocab := make([]rune, 0, len(set))
	for r := range set {
		vocab = append(vocab, r)
	}
	sort.Slice(vocab, func(i, j int) bool { return vocab[i] < vocab[j] })
	t := &Tokenizer{Vocab: vocab, ToID: make(map[rune]int, len(vocab))}
	for i, r := range vocab {
		t.ToID[r] = i
	}
	return t
}

func New(vocab []rune) *Tokenizer {
	t := &Tokenizer{Vocab: vocab, ToID: make(map[rune]int, len(vocab))}
	for i, r := range vocab {
		t.ToID[r] = i
	}
	return t
}

func (t *Tokenizer) Encode(s []rune) ([]int, error) {
	ids := make([]int, len(s))
	for i, r := range s {
		id, ok := t.ToID[r]
		if !ok {
			return nil, fmt.Errorf("character %q is not in the vocabulary", r)
		}
		ids[i] = id
	}
	return ids, nil
}

func (t *Tokenizer) Decode(ids []int) []rune {
	out := make([]rune, len(ids))
	for i, id := range ids {
		out[i] = t.Vocab[id]
	}
	return out
}
