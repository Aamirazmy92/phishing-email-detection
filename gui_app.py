import os
import re
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import joblib

from explain import explain
from preprocessing import preprocess_text

MODEL_PATH = os.path.join("models", "best_model.joblib")
VECTORIZER_PATH = os.path.join("models", "vectorizer.joblib")

PHISH_BG, PHISH_FG = "#fde8e8", "#b42318"
LEGIT_BG, LEGIT_FG = "#e7f6ec", "#1a7f37"
WORD_PHISH, WORD_LEGIT = "#f8b4b4", "#a7e3b8"

SAMPLES = {
    "Phishing sample": (
        "Security alert: unusual sign-in to your account. Verify your password "
        "within 24 hours or your mailbox will be suspended.\n"
        "http://secure-login.example.com/verify"
    ),
    "Legitimate sample": (
        "Hi team, attaching the notes from yesterday's meeting. Let me know if "
        "you have any questions. Thanks, Sam"
    ),
}


def load_artifacts():
    """Return (model, vectorizer) or (None, None) if not yet trained."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


def classify(text, model, vectorizer):
    """Classify raw email text. Returns 'Phishing' or 'Legitimate'."""
    if hasattr(vectorizer, "transform_raw"):
        X = vectorizer.transform_raw("", text)
    else:
        X = vectorizer.transform([preprocess_text(text)])
    pred = model.predict(X)[0]
    return "Phishing" if int(pred) == 1 else "Legitimate"


class EmailScannerApp:
    def __init__(self, root, model, vectorizer):
        self.root = root
        self.model = model
        self.vectorizer = vectorizer
        root.title("Phishing Email Scanner")
        root.geometry("900x560")
        root.minsize(760, 480)
        self._style()

        header = ttk.Frame(root, padding=(16, 12, 16, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Phishing Email Scanner", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Linear model on TF-IDF text features. Highlighted words show what drove each verdict.",
            style="Muted.TLabel",
        ).pack(anchor="w")

        self.paste_tab = ttk.Frame(root, padding=(16, 8, 16, 16))
        self.paste_tab.pack(fill="both", expand=True)
        self._build_paste_tab()

    # ---------- styling ----------
    def _style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Muted.TLabel", foreground="#667085")
        style.configure("Heading.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 6))

    # ---------- paste tab ----------
    def _build_paste_tab(self):
        tab = self.paste_tab
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)

        ttk.Label(tab, text="Paste the email (subject and body)", style="Heading.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        self.email_box = scrolledtext.ScrolledText(
            tab, wrap="word", font=("Segoe UI", 10), height=14, relief="solid", borderwidth=1)
        self.email_box.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        self.email_box.tag_configure("phish", background=WORD_PHISH)
        self.email_box.tag_configure("legit", background=WORD_LEGIT)

        buttons = ttk.Frame(tab)
        buttons.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Classify", style="Accent.TButton",
                   command=self.classify_pasted).pack(side="left")
        ttk.Button(buttons, text="Clear", command=self.clear_paste).pack(side="left", padx=6)
        ttk.Label(buttons, text="Try:", style="Muted.TLabel").pack(side="left", padx=(14, 4))
        for name, text in SAMPLES.items():
            ttk.Button(buttons, text=name, command=lambda t=text: self.load_sample(t)).pack(
                side="left", padx=2)

        result = ttk.Frame(tab)
        result.grid(row=0, column=1, rowspan=3, sticky="nsew")
        result.columnconfigure(0, weight=1)
        ttk.Label(result, text="Result", style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.verdict = tk.Label(result, text="Nothing checked yet", font=("Segoe UI", 16, "bold"),
                                bg="#f2f4f7", fg="#667085", pady=18)
        self.verdict.grid(row=1, column=0, sticky="ew")
        self.confidence_label = ttk.Label(result, text="", style="Muted.TLabel")
        self.confidence_label.grid(row=2, column=0, sticky="w", pady=(8, 2))
        self.confidence_bar = ttk.Progressbar(result, maximum=100)
        self.confidence_bar.grid(row=3, column=0, sticky="ew")
        ttk.Label(result, text="Pushes toward phishing", style="Heading.TLabel").grid(
            row=4, column=0, sticky="w", pady=(14, 2))
        self.phish_terms = tk.Label(result, text="-", anchor="w", justify="left", wraplength=290,
                                    fg=PHISH_FG)
        self.phish_terms.grid(row=5, column=0, sticky="ew")
        ttk.Label(result, text="Pushes toward legitimate", style="Heading.TLabel").grid(
            row=6, column=0, sticky="w", pady=(10, 2))
        self.legit_terms = tk.Label(result, text="-", anchor="w", justify="left", wraplength=290,
                                    fg=LEGIT_FG)
        self.legit_terms.grid(row=7, column=0, sticky="ew")
        ttk.Label(
            result,
            text="Confidence is distance from the decision boundary, not a calibrated probability. "
                 "Treat the verdict as a hint, not proof.",
            style="Muted.TLabel", wraplength=290, justify="left",
        ).grid(row=8, column=0, sticky="w", pady=(18, 0))

    def load_sample(self, text):
        self.email_box.delete("1.0", tk.END)
        self.email_box.insert("1.0", text)
        self.classify_pasted()

    def clear_paste(self):
        self.email_box.delete("1.0", tk.END)
        self._show_result(None)

    def classify_pasted(self):
        text = self.email_box.get("1.0", tk.END).strip()
        if not text:
            self._show_result(None, "Paste some email text first")
            return
        self._show_result(explain(text, self.model, self.vectorizer))

    def _show_result(self, result, message="Nothing checked yet"):
        for tag in ("phish", "legit"):
            self.email_box.tag_remove(tag, "1.0", tk.END)
        if result is None:
            self.verdict.config(text=message, bg="#f2f4f7", fg="#667085")
            self.confidence_label.config(text="")
            self.confidence_bar["value"] = 0
            self.phish_terms.config(text="-")
            self.legit_terms.config(text="-")
            return
        phishing = result["label"] == "Phishing"
        self.verdict.config(text=("⚠  " if phishing else "✓  ") + result["label"],
                            bg=PHISH_BG if phishing else LEGIT_BG,
                            fg=PHISH_FG if phishing else LEGIT_FG)
        self.confidence_label.config(text=f"Confidence {result['confidence']:.0%}")
        self.confidence_bar["value"] = result["confidence"] * 100
        self.phish_terms.config(text=", ".join(w for w, _ in result["phishing_terms"]) or "-")
        self.legit_terms.config(text=", ".join(w for w, _ in result["legit_terms"]) or "-")
        self._highlight(result)

    def _highlight(self, result):
        for tag, terms in (("phish", result["phishing_terms"]), ("legit", result["legit_terms"])):
            for word, _ in terms:
                if not re.fullmatch(r"[a-z]+", word):  # skip bigrams and named signals
                    continue
                start = "1.0"
                while True:
                    pos = self.email_box.search(rf"\m{word}\M", start, tk.END, nocase=True, regexp=True)
                    if not pos:
                        break
                    end = f"{pos}+{len(word)}c"
                    self.email_box.tag_add(tag, pos, end)
                    start = end


def main():
    model, vectorizer = load_artifacts()
    if model is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Model not found",
            "No trained model found. Please run 'py train_multisource.py' (or "
            "'py train_evaluate.py') first.",
        )
        return
    root = tk.Tk()
    EmailScannerApp(root, model, vectorizer)
    root.mainloop()


if __name__ == "__main__":
    main()
