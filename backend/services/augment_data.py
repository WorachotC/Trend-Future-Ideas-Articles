import json
import os

INPUT_FILE = "data/processed/jenosize_train_data.jsonl"
OUTPUT_FILE = "data/processed/jenosize_augmented.jsonl"

def split_text(text, chunk_size=1000, overlap=100):
    """
    Splits long text into smaller chunks with overlap to maintain context.
    
    Args:
        text (str): The original article text.
        chunk_size (int): The maximum size of each text chunk.
        overlap (int): The number of characters to overlap between chunks.
    
    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # If we are not at the very end of the text, try to find a natural split point
        if end < text_len:
            # Find the last space within the chunk limit to avoid cutting words in half
            last_space = text[start:end].rfind(' ')
            if last_space != -1:
                end = start + last_space
        
        # Extract the chunk and remove leading/trailing whitespace
        chunk = text[start:end].strip()

        # Only keep chunks that have substantial content (> 200 characters)
        # This filters out small fragments or empty lines
        if len(chunk) > 200: 
            chunks.append(chunk)
        
        # Move the start pointer forward, subtracting overlap
        # This ensures the next chunk contains some context from the previous one
        start = end - overlap

    return chunks

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    augmented_dataset = []
    original_count = 0

    print(f"🔄 Processing file: {INPUT_FILE}...")

    try:
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"Input file not found at: {INPUT_FILE}")

        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue # Skip empty lines

                entry = json.loads(line)
                original_text = entry.get('output', '')
                title = entry.get('input', '')
                instruction = entry.get('instruction', '')
                
                original_count += 1

                # Condition: If the article is long (> 1500 chars), split it.
                if len(original_text) > 1500:
                    chunks = split_text(original_text)
                    
                    # Create new training examples from chunks
                    for i, chunk in enumerate(chunks):
                        new_entry = {
                            "instruction": instruction,
                            "input": f"{title} (Part {i+1})", # Append part number to context
                            "output": chunk
                        }
                        augmented_dataset.append(new_entry)
                else:
                    # If the text is short enough, keep it as is
                    augmented_dataset.append(entry)

        # Save the new augmented dataset to a file
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in augmented_dataset:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print("-" * 40)
        print(f"✅ Original Articles: {original_count}")
        print(f"🚀 Augmented Examples: {len(augmented_dataset)}")
        print(f"📂 Output Saved to: {OUTPUT_FILE}")
        print("-" * 40)
        print("👉 You can now upload this file to Colab for training!")

    except Exception as e:
        print(f"❌ Error: {e}")