"""
Quick interactive Marathi chat with the (base + adapter) model.

Run: python -m src.inference --adapter tusharislampure29/qwen2.5-1.5b-marathi-instruct
"""

from __future__ import annotations

import argparse

MARATHI_SYSTEM = "तुम्ही एक उपयुक्त AI सहाय्यक आहात जो मराठीत स्पष्ट आणि अचूक उत्तरे देतो."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="HF adapter repo ID, optional.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model}" + (f" + {args.adapter}" if args.adapter else ""))
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    history = [{"role": "system", "content": MARATHI_SYSTEM}]
    print("Type your prompt in Marathi. Empty line to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        history.append({"role": "user", "content": user})
        text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": gen})
        print(f"\nAssistant: {gen}\n")


if __name__ == "__main__":
    main()
