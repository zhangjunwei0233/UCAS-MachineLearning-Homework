import argparse
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interactive sentiment inference with a fine-tuned model.")
    parser.add_argument(
        "--model_dir",
        default="project-sentiment-analysis/model-output",
        help="Path to the saved model directory (containing config, tokenizer, and weights).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    id2label = model.config.id2label or {}

    print("Interactive sentiment demo. Type 'quit' or press Ctrl+C to exit.")
    while True:
        try:
            text = input("\nEnter a phrase: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except EOFError:
            print("\nExiting.")
            break

        if not text or text.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_id = int(torch.argmax(probs).item())
            pred_label = id2label.get(pred_id, str(pred_id))

        prob_percent = float(probs[pred_id].item() * 100)
        print(f"Prediction: {pred_label} (confidence: {prob_percent:.1f}%)")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # Broad catch to provide a helpful message before exiting.
        print(f"Error during inference: {err}", file=sys.stderr)
        sys.exit(1)
