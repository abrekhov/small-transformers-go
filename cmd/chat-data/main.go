package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type topic struct {
	Topic      string   `json:"topic"`
	Questions  []string `json:"questions"`
	Validation []string `json:"validation"`
	Personal   string   `json:"personal"`
	Test       string   `json:"test"`
	Answers    []string `json:"answers"`
}

type additions struct {
	Train      []string `json:"train"`
	Validation []string `json:"validation"`
}

type example struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

func main() {
	source := flag.String("source", "datasets/chat-topics.json", "authored topic seeds")
	questions := flag.String("questions", "datasets/chat-questions.json", "natural-language training and validation questions")
	out := flag.String("out", "data", "output directory")
	flag.Parse()
	if err := generate(*source, *questions, *out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func generate(source, questions, out string) error {
	raw, err := os.ReadFile(source)
	if err != nil {
		return err
	}
	var topics []topic
	if err := json.Unmarshal(raw, &topics); err != nil {
		return err
	}
	raw, err = os.ReadFile(questions)
	if err != nil {
		return err
	}
	var extra map[string]additions
	if err := json.Unmarshal(raw, &extra); err != nil {
		return err
	}
	sets := make([][]example, 3)
	seen := make(map[string]int)
	for _, t := range topics {
		train := t.Questions
		if len(train) == 0 {
			for _, format := range []string{"Tell me about %s.", "Speak to me of %s.", "What dost thou think of %s?", "Give me thy thoughts on %s.", "I wish to understand %s.", "Offer me counsel about %s."} {
				train = append(train, fmt.Sprintf(format, t.Topic))
			}
		}
		if t.Personal != "" {
			train = append(train, t.Personal)
		}
		train = append(train, extra[t.Topic].Train...)
		validation := t.Validation
		if len(validation) == 0 {
			validation = []string{fmt.Sprintf("Couldst thou help me understand %s?", t.Topic), fmt.Sprintf("Share a gentle thought concerning %s.", t.Topic)}
		}
		validation = append(validation, extra[t.Topic].Validation...)
		var framed []string
		for _, question := range train {
			for _, text := range []string{question, strings.ToLower(question)} {
				for _, prefix := range []string{"", "Dear bard, ", "My friend, "} {
					for _, suffix := range []string{"", " Please.", " What wouldst thou say?"} {
						framed = append(framed, prefix+text+suffix)
					}
				}
			}
		}
		for split, questions := range [][]string{framed, validation, {t.Test}} {
			for _, question := range questions {
				key := strings.ToLower(strings.Join(strings.Fields(question), " "))
				if key == "" || len(t.Answers) == 0 {
					return fmt.Errorf("empty question or answers for %s", t.Topic)
				}
				if previous, exists := seen[key]; exists && previous != split {
					return fmt.Errorf("question appears in multiple splits: %s", question)
				}
				seen[key] = split
				for _, answer := range t.Answers {
					length := len([]rune("\nUSER:\n"+question+"\nBARD:\n"+answer+"\nEND\n")) - 1
					if length > 256 {
						if split != 0 {
							return fmt.Errorf("held-out question too long for context 256: %s", question)
						}
						continue
					}
					sets[split] = append(sets[split], example{question, answer})
				}
			}
		}
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	for i, name := range []string{"train", "validation", "test"} {
		path := filepath.Join(out, "shakespeare-chat-"+name+".jsonl")
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		encoder := json.NewEncoder(f)
		for _, row := range sets[i] {
			if err := encoder.Encode(row); err != nil {
				f.Close()
				return err
			}
		}
		if err := f.Close(); err != nil {
			return err
		}
		fmt.Printf("%s: %d examples\n", path, len(sets[i]))
	}
	return nil
}
