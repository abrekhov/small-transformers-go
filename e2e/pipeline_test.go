package e2e_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var bin, workDir, corpusPath, modelPath string

const corpusText = `KING RICHARD:
The weary sun hath made a golden set,
And by the bright track of his fiery car
Gives token of a goodly day tomorrow.

QUEEN MARGARET:
What, are you leaving? Stay, I prithee, stay!
The night is dark, and honour is unhorsed.
We will attend upon the morning bell,
And crown our patience with a cheerful dawn.

DUKE OF YORK:
So rises wrath, and so the tempest swells,
Yet love, once planted, outlasts every storm.

`

var _ = BeforeSuite(func() {
	workDir = GinkgoT().TempDir()
	bin = filepath.Join(workDir, "small-transformers-go")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "build", "-o", bin, ".")
	cmd.Dir = ".."
	out, err := cmd.CombinedOutput()
	Expect(err).NotTo(HaveOccurred(), string(out))
	corpusPath = filepath.Join(workDir, "data", "corpus.txt")
	modelPath = filepath.Join(workDir, "checkpoints", "model.bin")
})

var _ = Describe("download → train → checkpoint → inference", Ordered, func() {
	It("downloads a corpus over HTTP", func(ctx SpecContext) {
		corpus := strings.Repeat(corpusText, 40)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, corpus)
		}))
		defer server.Close()
		cmd := exec.CommandContext(ctx, bin, "download", "-url", server.URL, "-out", corpusPath)
		out, err := cmd.CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		raw, err := os.ReadFile(corpusPath)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(raw)).To(Equal(corpus))
	}, SpecTimeout(10*time.Second))

	It("preserves an existing corpus when the server fails", func(ctx SpecContext) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer server.Close()
		cmd := exec.CommandContext(ctx, bin, "download", "-url", server.URL, "-out", corpusPath)
		out, err := cmd.CombinedOutput()
		Expect(err).To(HaveOccurred())
		Expect(string(out)).To(ContainSubstring("HTTP 503"))
		raw, err := os.ReadFile(corpusPath)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(raw)).To(Equal(strings.Repeat(corpusText, 40)))
	}, SpecTimeout(10*time.Second))

	It("trains the downloaded corpus and decreases loss", func(ctx SpecContext) {
		cmd := exec.CommandContext(ctx, bin, "train",
			"-data", corpusPath, "-out", modelPath,
			"-steps", "80", "-batch", "8", "-seq", "32",
			"-dim", "32", "-heads", "2", "-layers", "2",
			"-eval-every", "20", "-seed", "42")
		out, err := cmd.CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		fmt.Fprint(GinkgoWriter, string(out))
		losses := parseLosses(string(out))
		Expect(losses).To(HaveLen(5))
		Expect(losses[len(losses)-1]).To(BeNumerically("<", losses[0]-0.5))
		_, err = os.Stat(modelPath)
		Expect(err).NotTo(HaveOccurred())
	}, SpecTimeout(120*time.Second))

	It("evaluates a saved checkpoint reproducibly on held-out text", func(ctx SpecContext) {
		args := []string{"eval", "-model", modelPath, "-data", corpusPath, "-batch", "8"}
		out, err := exec.CommandContext(ctx, bin, args...).CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		Expect(string(out)).To(ContainSubstring("held-out chars"))
		Expect(parseLosses(string(out))[0]).To(BeNumerically("<", 3.3))
		again, err := exec.CommandContext(ctx, bin, args...).CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(again))
		Expect(again).To(Equal(out))
	}, SpecTimeout(30*time.Second))

	It("scans the complete validation tail including its partial final window", func(ctx SpecContext) {
		out, err := exec.CommandContext(ctx, bin, "eval", "-model", modelPath, "-data", corpusPath, "-batches", "0").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		chars := len([]rune(strings.Repeat(corpusText, 40)))
		validationChars := chars - int(float64(chars)*.9)
		Expect(string(out)).To(ContainSubstring(fmt.Sprintf("%d predictions; full scan", validationChars-1)))
		Expect(parseLosses(string(out))[0]).To(BeNumerically("<", 3.3))
	}, SpecTimeout(30*time.Second))

	It("warm-starts a checkpoint without changing its source file", func(ctx SpecContext) {
		before, err := os.ReadFile(modelPath)
		Expect(err).NotTo(HaveOccurred())
		warm := filepath.Join(workDir, "warm.bin")
		out, err := exec.CommandContext(ctx, bin, "train", "-data", corpusPath,
			"-init", modelPath, "-out", warm, "-steps", "80", "-batch", "8", "-lr", ".0005", "-eval-every", "80", "-seed", "43").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		Expect(string(out)).To(ContainSubstring("optimizer reset"))
		Expect(string(out)).To(ContainSubstring("dim=32 heads=2 layers=2 ctx=32"))
		after, err := os.ReadFile(modelPath)
		Expect(err).NotTo(HaveOccurred())
		Expect(after).To(Equal(before))
		initial, err := exec.CommandContext(ctx, bin, "eval", "-model", modelPath, "-data", corpusPath, "-batch", "8").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(initial))
		improved, err := exec.CommandContext(ctx, bin, "eval", "-model", warm, "-data", corpusPath, "-batch", "8").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(improved))
		Expect(parseLosses(string(improved))[0]).To(BeNumerically("<", parseLosses(string(initial))[0]-.05))
	}, SpecTimeout(60*time.Second))

	It("reloads the checkpoint and generates exactly N vocabulary characters reproducibly", func(ctx SpecContext) {
		args := []string{"generate", "-model", modelPath, "-prompt", "KING", "-n", "120", "-temp", "0.8", "-topk", "0", "-seed", "7"}
		out, err := exec.CommandContext(ctx, bin, args...).Output()
		Expect(err).NotTo(HaveOccurred())
		Expect(string(out)).To(HavePrefix("KING"))
		Expect([]rune(string(out))).To(HaveLen(124))
		for _, r := range string(out) {
			Expect(strings.ContainsRune(corpusText, r)).To(BeTrue())
		}
		again, err := exec.CommandContext(ctx, bin, args...).Output()
		Expect(err).NotTo(HaveOccurred())
		Expect(again).To(Equal(out))
	}, SpecTimeout(30*time.Second))

	It("handles a positive temperature near float32 underflow", func(ctx SpecContext) {
		out, err := exec.CommandContext(ctx, bin, "generate", "-model", modelPath, "-prompt", "KING", "-n", "4", "-temp", "1e-40").Output()
		Expect(err).NotTo(HaveOccurred())
		Expect([]rune(string(out))).To(HaveLen(8))
	}, SpecTimeout(10*time.Second))

	DescribeTable("rejects invalid inference requests without a panic", func(extra []string, expected string) {
		args := append([]string{"generate", "-model", modelPath}, extra...)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, bin, args...).CombinedOutput()
		Expect(err).To(HaveOccurred())
		Expect(string(out)).To(ContainSubstring(expected))
		Expect(string(out)).NotTo(ContainSubstring("panic:"))
	},
		Entry("unknown prompt character", []string{"-prompt", "☃"}, "not in the vocabulary"),
		Entry("empty prompt", []string{"-prompt", ""}, "prompt must be nonempty"),
		Entry("zero temperature", []string{"-temp", "0"}, "temp positive"),
		Entry("nonfinite temperature", []string{"-temp", "NaN"}, "temp positive"),
		Entry("negative token count", []string{"-n", "-1"}, "n and topk nonnegative"),
	)
})

var _ = Describe("training boundaries", func() {
	It("trains when the corpus has exactly seq+1 characters", func(ctx SpecContext) {
		path := filepath.Join(workDir, "minimal.txt")
		Expect(os.WriteFile(path, []byte("abcde"), 0o644)).To(Succeed())
		out, err := exec.CommandContext(ctx, bin, "train", "-data", path,
			"-out", filepath.Join(workDir, "minimal.bin"), "-seq", "4", "-batch", "1",
			"-steps", "1", "-dim", "8", "-heads", "1", "-layers", "1", "-split", "1").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		Expect(string(out)).To(ContainSubstring("saved checkpoint"))
	}, SpecTimeout(10*time.Second))

	DescribeTable("rejects invalid training settings without a panic", func(extra []string) {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, bin, append([]string{"train"}, extra...)...).CombinedOutput()
		Expect(err).To(HaveOccurred())
		Expect(string(out)).NotTo(ContainSubstring("panic:"))
	},
		Entry("zero batch", []string{"-batch", "0"}),
		Entry("zero context", []string{"-seq", "0"}),
		Entry("zero heads", []string{"-heads", "0"}),
		Entry("dimension not divisible by heads", []string{"-dim", "31", "-heads", "2"}),
		Entry("zero log interval", []string{"-eval-every", "0"}),
		Entry("nonfinite learning rate", []string{"-lr", "NaN"}),
		Entry("zero evaluation batches", []string{"-eval-batches", "0"}),
	)

	It("keeps validation out of training and saves the best rather than last checkpoint", func(ctx SpecContext) {
		path := filepath.Join(workDir, "split.txt")
		checkpoint := filepath.Join(workDir, "split.bin")
		Expect(os.WriteFile(path, []byte(strings.Repeat("a", 900)+strings.Repeat("b", 100)), 0o644)).To(Succeed())
		out, err := exec.CommandContext(ctx, bin, "train", "-data", path, "-out", checkpoint,
			"-steps", "100", "-batch", "4", "-seq", "16", "-dim", "8", "-heads", "1", "-layers", "1", "-lr", ".01", "-eval-every", "20").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		Expect(string(out)).To(ContainSubstring("train chars: 900, validation chars: 100"))
		losses := parseLosses(string(out))
		Expect(losses[len(losses)-1]).To(BeNumerically("<", .15))
		matches := regexp.MustCompile(`val ([0-9.]+)`).FindAllStringSubmatch(string(out), -1)
		Expect(matches).To(HaveLen(6))
		best, last := 100.0, 0.0
		for _, match := range matches {
			last, err = strconv.ParseFloat(match[1], 64)
			Expect(err).NotTo(HaveOccurred())
			best = min(best, last)
		}
		Expect(last).To(BeNumerically(">", best+1))
		evalOut, err := exec.CommandContext(ctx, bin, "eval", "-model", checkpoint, "-data", path, "-batch", "4").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(evalOut))
		Expect(parseLosses(string(evalOut))[0]).To(BeNumerically("~", best, .0001))
	}, SpecTimeout(30*time.Second))

	It("reports a truncated checkpoint without a panic", func(ctx SpecContext) {
		path := filepath.Join(workDir, "broken.bin")
		Expect(os.WriteFile(path, []byte("broken"), 0o644)).To(Succeed())
		out, err := exec.CommandContext(ctx, bin, "generate", "-model", path).CombinedOutput()
		Expect(err).To(HaveOccurred())
		Expect(string(out)).NotTo(ContainSubstring("panic:"))
	}, SpecTimeout(10*time.Second))

	It("learns to recall a cue across intervening tokens", func(ctx SpecContext) {
		path := filepath.Join(workDir, "recall.txt")
		checkpoint := filepath.Join(workDir, "recall.bin")
		Expect(os.WriteFile(path, []byte(strings.Repeat("a.......a\nb.......b\n", 400)), 0o644)).To(Succeed())
		out, err := exec.CommandContext(ctx, bin, "train", "-data", path, "-out", checkpoint,
			"-steps", "1200", "-batch", "8", "-seq", "16", "-dim", "32", "-heads", "2", "-layers", "1", "-lr", ".003", "-split", "1").CombinedOutput()
		Expect(err).NotTo(HaveOccurred(), string(out))
		for _, cue := range []string{"a", "b"} {
			prompt := cue + "......."
			text, err := exec.CommandContext(ctx, bin, "generate", "-model", checkpoint, "-prompt", prompt, "-n", "1", "-topk", "1").Output()
			Expect(err).NotTo(HaveOccurred())
			Expect(string(text)).To(Equal(prompt + cue))
		}
	}, SpecTimeout(120*time.Second))
})

var lossRe = regexp.MustCompile(`loss ([0-9.]+)`)

func parseLosses(out string) []float64 {
	var losses []float64
	for _, m := range lossRe.FindAllStringSubmatch(out, -1) {
		v, err := strconv.ParseFloat(m[1], 64)
		Expect(err).NotTo(HaveOccurred())
		losses = append(losses, v)
	}
	return losses
}
