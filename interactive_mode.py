# modified code from ChatGPT (4o, paid version, November 26, 2024)
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# SETTINGS
MODEL_DIR = "model_output"  # Path to our fine-tuned/trained model

# Load the fine-tuned model and tokenizer
print("Loading fine-tuned model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

def generate_text(prompt, word_count):
    """
    Generates text based on the provided prompt and desired word count.
    """
    # Tokenize the prompt with padding and truncation
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate text with proper attention mask and padding token ID
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=word_count,  # Assume word_count translates roughly to tokens
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id  # Explicitly set the pad token
    )

    # Decode the generated tokens and return the text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def interactive_prompt():
    """
    This function sets up the interactive terminal.
    """
    print("Enter the narrative prompt (type 'x' to quit).")
    while True:
        try:
            # ask for writing prompt
            scenario = input("\nEnter a scenario: ")
            if not scenario:
                print("Scenario cannot be empty. Please try again.")
                continue
            if scenario.lower() == "x":
                print("Exiting interactive mode.")
                break
                
            # add instructions for the model to interpret
            prompt = f"Write a cohesive story about the following scenario: {scenario}"

            # ask for word count
            word_count_input = input("Enter the desired word count (default is 50): \n")
            try:
                if word_count_input:
                    word_count = int(word_count_input)
                    if word_count <= 0:
                        print("Word count must be a positive integer. Using default of 50.")
                        word_count = 50
                    if word_count > 2000:
                        print("Word count is too large! Using default of 50.")
                        word_count = 50
                else:
                    word_count = 50
            except ValueError:
                print("Invalid word count entered. Using default of 50.")
                word_count = 50

            # generate the response within the specified word count
            generated_text = generate_text(prompt, word_count=word_count)
            print(f"\nGenerated Narrative:\n{generated_text}")
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break

if __name__ == "__main__":
    interactive_prompt()