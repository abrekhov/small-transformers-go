package e2e_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const chatTraining = `{"question":"Hello","answer":"Good morrow, friend."}
{"question":"Hello friend","answer":"Good morrow, friend."}
{"question":"Hi friend","answer":"Good morrow, friend."}
{"question":"Goodbye","answer":"Fare thee well, friend."}
{"question":"Goodbye friend","answer":"Fare thee well, friend."}
{"question":"Bye friend","answer":"Fare thee well, friend."}
`

const chatValidation = `{"question":"Hello good friend","answer":"Good morrow, friend."}
{"question":"Goodbye good friend","answer":"Fare thee well, friend."}
`

var _ = Describe("conversational fine-tuning", Ordered, func() {
	var trainPath, validationPath, base, model string

	BeforeAll(func() {
		trainPath = filepath.Join(workDir, "chat-train.jsonl")
		validationPath = filepath.Join(workDir, "chat-validation.jsonl")
		base = filepath.Join(workDir, "chat-base.bin")
		model = filepath.Join(workDir, "chat-model.bin")
		Expect(os.WriteFile(trainPath, []byte(chatTraining), 0o644)).To(Succeed())
		Expect(os.WriteFile(validationPath, []byte(chatValidation), 0o644)).To(Succeed())
	})

	It("fine-tunes base weights with an expanded vocabulary and answers-only loss", func(ctx SpecContext) {
		corpus := filepath.Join(workDir, "chat-base.txt")
		Expect(os.WriteFile(corpus, []byte(strings.Repeat("Hello friend. Goodbye friend.\n", 40)), 0o644)).To(Succeed())
		out, err := exec.CommandContext(ctx, bin, "train", "-data", corpus, "-out", base,
			"-steps", "5", "-batch", "4", "-seq", "64", "-dim", "24", "-heads", "2", "-layers", "1").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		original, err := os.ReadFile(base)
		Expect(err).NotTo(HaveOccurred())
		out, err = exec.CommandContext(ctx, bin, "train", "-chat", "-data", trainPath, "-validation", validationPath,
			"-init", base, "-out", model, "-steps", "2000", "-batch", "4", "-lr", ".003", "-eval-every", "250", "-eval-batches", "2").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		Expect(string(out)).To(ContainSubstring("answer-only loss; vocab: 96"))
		unchanged, err := os.ReadFile(base)
		Expect(err).NotTo(HaveOccurred())
		Expect(unchanged).To(Equal(original))
		out, err = exec.CommandContext(ctx, bin, "eval", "-chat", "-model", model, "-data", validationPath, "-batch", "2", "-batches", "0").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		Expect(parseLosses(string(out))[0]).To(BeNumerically("<", .15))
	}, SpecTimeout(120*time.Second))

	It("answers unseen-length questions differently and stops before END", func(ctx SpecContext) {
		for _, example := range []struct{ question, answer string }{
			{"Hello good friend", "Good morrow, friend."},
			{"Goodbye good friend", "Fare thee well, friend."},
		} {
			out, err := exec.CommandContext(ctx, bin, "chat", "-model", model, "-prompt", example.question, "-n", "60", "-topk", "1").Output()
			Expect(err).NotTo(HaveOccurred())
			Expect(strings.TrimSpace(string(out))).To(Equal(example.answer))
		}
	}, SpecTimeout(15*time.Second))

	It("accepts repeated messages on stdin and exits on quit", func(ctx SpecContext) {
		cmd := exec.CommandContext(ctx, bin, "chat", "-model", model, "-n", "60", "-topk", "1")
		cmd.Stdin = strings.NewReader("Hello good friend\nGoodbye good friend\n/quit\nHello\n")
		out, err := cmd.CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		Expect(string(out)).To(Equal("Good morrow, friend.\nFare thee well, friend.\n"))
	}, SpecTimeout(15*time.Second))

	It("rejects questions that leak between training and validation", func(ctx SpecContext) {
		out, err := exec.CommandContext(ctx, bin, "train", "-chat", "-data", trainPath, "-validation", trainPath,
			"-init", base, "-out", filepath.Join(workDir, "leaked.bin"), "-steps", "1").CombinedOutput()
		Expect(err).To(HaveOccurred())
		Expect(string(out)).To(ContainSubstring("also appears in training"))
		Expect(string(out)).NotTo(ContainSubstring("panic:"))
	}, SpecTimeout(10*time.Second))

	It("does not pretend a completion-only checkpoint is chat-trained", func(ctx SpecContext) {
		out, err := exec.CommandContext(ctx, bin, "chat", "-model", base, "-prompt", "Hello").CombinedOutput()
		Expect(err).To(HaveOccurred())
		Expect(string(out)).To(ContainSubstring("not chat-trained"))
	}, SpecTimeout(10*time.Second))
})
