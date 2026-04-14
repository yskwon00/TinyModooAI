import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

def train_tokenizer():
    print("Loading dataset for tokenizer training...")
    # Using wikitext as a base, you can add your own local files here
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Optional: Save dataset to text file for training
    batch_size = 1000
    def batch_iterator():
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]["text"]

    # Initialize Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Initialize Trainer
    trainer = BpeTrainer(
        vocab_size=32000,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<s>", "</s>"],
        show_progress=True
    )

    # Train
    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Wrap in Transformers class
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token="<s>",
        eos_token="</s>",
    )

    # Save
    output_dir = "tinymodoo_tokenizer"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fast_tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

if __name__ == "__main__":
    train_tokenizer()
