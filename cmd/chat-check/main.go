package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

type example struct{ Question, Answer string }
type group struct {
	Question string
	Answers  []string
}
type result struct {
	Question            string  `json:"question"`
	Reply               string  `json:"reply"`
	ReferenceScore      float64 `json:"reference_word_f1"`
	ReferenceTopicMatch bool    `json:"reference_topic_match"`
	Error               string  `json:"error,omitempty"`
}

var words = regexp.MustCompile(`[a-z]+`)

func score(a, b string) float64 {
	x, y := words.FindAllString(strings.ToLower(a), -1), words.FindAllString(strings.ToLower(b), -1)
	counts := map[string]int{}
	for _, word := range y {
		counts[word]++
	}
	common := 0
	for _, word := range x {
		if counts[word] > 0 {
			common++
			counts[word]--
		}
	}
	if len(x)+len(y) == 0 {
		return 0
	}
	return 2 * float64(common) / float64(len(x)+len(y))
}

func main() {
	file, err := os.Open(os.Args[3])
	if err != nil {
		panic(err)
	}
	defer file.Close()
	var groups []group
	indices := map[string]int{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var e example
		if err := json.Unmarshal(scanner.Bytes(), &e); err != nil {
			panic(err)
		}
		i, exists := indices[e.Question]
		if !exists {
			i = len(groups)
			indices[e.Question] = i
			groups = append(groups, group{Question: e.Question})
		}
		groups[i].Answers = append(groups[i].Answers, e.Answer)
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	out, err := os.Create(os.Args[4])
	if err != nil {
		panic(err)
	}
	defer out.Close()
	encoder := json.NewEncoder(out)
	correct := 0
	for i, g := range groups {
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		cmd := exec.CommandContext(ctx, os.Args[1], "chat", "-model", os.Args[2], "-prompt", g.Question, "-n", "200", "-temp", ".7", "-topk", "20", "-seed", "7")
		raw, err := cmd.CombinedOutput()
		cancel()
		r := result{Question: g.Question, Reply: strings.TrimSpace(string(raw))}
		if err != nil {
			r.Error = err.Error()
		}
		best, bestGroup := -1.0, -1
		for j, candidate := range groups {
			for _, answer := range candidate.Answers {
				s := score(r.Reply, answer)
				if j == i && s > r.ReferenceScore {
					r.ReferenceScore = s
				}
				if s > best {
					best = s
					bestGroup = j
				}
			}
		}
		r.ReferenceTopicMatch = err == nil && bestGroup == i && r.ReferenceScore >= .65 && !strings.Contains(r.Reply, "END")
		if r.ReferenceTopicMatch {
			correct++
		}
		if err := encoder.Encode(r); err != nil {
			panic(err)
		}
		fmt.Printf("%d/%d topic_match=%v reference_f1=%.2f question=%s\n", i+1, len(groups), r.ReferenceTopicMatch, r.ReferenceScore, g.Question)
	}
	fmt.Printf("Reference-matching topical replies: %d/%d (%.1f%%). This lexical check is not a general semantic or fluency benchmark.\n", correct, len(groups), 100*float64(correct)/float64(len(groups)))
}
