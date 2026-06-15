import email
import imaplib
import os
import tkinter as tk
from email.header import decode_header
from tkinter import messagebox, scrolledtext

import joblib

from preprocessing import preprocess_text

MODEL_PATH = os.path.join("models", "best_model.joblib")
VECTORIZER_PATH = os.path.join("models", "vectorizer.joblib")


def load_artifacts():
    """Return (model, vectorizer) or (None, None) if not yet trained."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


def classify(text, model, vectorizer):
    """Classify raw email text. Returns 'Phishing' or 'Legitimate'."""
    cleaned = preprocess_text(text)
    pred = model.predict(vectorizer.transform([cleaned]))[0]
    return "Phishing" if int(pred) == 1 else "Legitimate"


class EmailScannerApp:
    def __init__(self, root, model, vectorizer):
        self.root = root
        self.model = model
        self.vectorizer = vectorizer
        root.title("Phishing Email Scanner")
        root.geometry("520x540")

        tk.Label(root, text="Phishing Email Scanner", font=("Arial", 16)).pack(pady=8)

        # --- Offline tester ---
        tk.Label(root, text="Paste an email to test (no login needed):").pack()
        self.email_box = scrolledtext.ScrolledText(root, width=60, height=8)
        self.email_box.pack(padx=10, pady=4)
        tk.Button(root, text="Classify pasted email",
                  command=self.classify_pasted).pack(pady=4)
        self.result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=4)

        # --- Gmail scan ---
        tk.Label(root, text="--- or scan a Gmail inbox ---").pack(pady=6)
        tk.Label(root, text="Email:").pack()
        self.email_entry = tk.Entry(root, width=40)
        self.email_entry.pack()
        tk.Label(root, text="App password:").pack()
        self.password_entry = tk.Entry(root, show="*", width=40)
        self.password_entry.pack()
        tk.Button(root, text="Login and Scan Inbox",
                  command=self.login_and_scan).pack(pady=8)

    def classify_pasted(self):
        text = self.email_box.get("1.0", tk.END).strip()
        if not text:
            self.result_label.config(text="Please paste some email text first.")
            return
        label = classify(text, self.model, self.vectorizer)
        colour = "red" if label == "Phishing" else "green"
        self.result_label.config(text=f"Prediction: {label}", fg=colour)

    def login_and_scan(self):
        email_user = self.email_entry.get()
        email_pass = self.password_entry.get()
        try:
            imap = imaplib.IMAP4_SSL("imap.gmail.com")
            imap.login(email_user, email_pass)
            imap.select("inbox")
            _, messages = imap.search(None, "UNSEEN")
            mail_ids = messages[0].split()
            results = []
            for num in mail_ids[:5]:
                _, data = imap.fetch(num, "(RFC822)")
                msg = email.message_from_bytes(data[0][1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8", errors="ignore")
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")
                label = classify(f"{subject} {body}", self.model, self.vectorizer)
                results.append(f"Subject: {subject}\nPrediction: {label}\n")
            imap.logout()
            self.show_results(results or ["No unread emails found."])
        except Exception as exc:
            messagebox.showerror("Login Failed", f"Error: {exc}")

    def show_results(self, results):
        win = tk.Toplevel(self.root)
        win.title("Scan Results")
        box = scrolledtext.ScrolledText(win, width=80, height=20)
        box.pack(padx=10, pady=10)
        box.insert(tk.END, "\n\n".join(results))
        box.config(state=tk.DISABLED)


def main():
    model, vectorizer = load_artifacts()
    if model is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Model not found",
            "No trained model found. Please run 'py train_evaluate.py' first.",
        )
        return
    root = tk.Tk()
    EmailScannerApp(root, model, vectorizer)
    root.mainloop()


if __name__ == "__main__":
    main()
