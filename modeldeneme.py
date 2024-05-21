import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the model and tokenizer
model_path = r"C:\Users\saren\Desktop\modeldeneme\distilbert_model_eng"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

model.eval()

# Load the CSV file
csv_path = r"C:\Users\saren\Desktop\bitirme\eng\jasmin.rs.csv"  # Update this to the path of your CSV file
df = pd.read_csv(csv_path, sep=';')

# Select the names column (update 'name_column' to your actual column name with names)
names = df["Name"].tolist()

# Output file
output_file_path = 'english.txt'

# Open a file to write the classification results
with open(output_file_path, 'w') as output_file:
    # Classify each name

    for name in names:
        # Tokenize the name
        try:

            inputs = tokenizer(name, return_tensors="pt", padding=True, truncation=True)

            # Get the model's logits
            with torch.no_grad():
                logits = model(**inputs).logits

            # Convert logits to probabilities and then to binary predictions
            predictions = torch.argmax(logits, dim=1).tolist()

            # Write the name and its classification to the text file
            classification = 'random' if predictions[0] == 0 else 'not random'
            output_file.write(f"{name}: {classification}\n")
        except:
            pass

print(f"Classification complete. Results saved to {output_file_path}")
