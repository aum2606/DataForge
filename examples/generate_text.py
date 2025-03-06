"""
Example script for generating synthetic text data.
"""

import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules directly
from data_types import text_data
from utils import exporters

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'text')
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Generate random words
    print("Generating random words...")
    words = [text_data.generate_random_word() for _ in range(10)]
    print(f"Generated words: {', '.join(words)}")
    print("\n" + "-"*80 + "\n")
    
    # Example 2: Generate random sentences
    print("Generating random sentences...")
    sentences = [text_data.generate_sentence() for _ in range(5)]
    for i, sentence in enumerate(sentences, 1):
        print(f"Sentence {i}: {sentence}")
    print("\n" + "-"*80 + "\n")
    
    # Example 3: Generate a paragraph
    print("Generating a paragraph...")
    paragraph = text_data.generate_paragraph()
    print(paragraph)
    print("\n" + "-"*80 + "\n")
    
    # Example 4: Generate an article
    print("Generating an article...")
    article = text_data.generate_article()
    print(article)
    print("\n" + "-"*80 + "\n")
    
    # Example 5: Generate a conversation
    print("Generating a conversation...")
    conversation = text_data.generate_conversation()
    print(conversation)
    print("\n" + "-"*80 + "\n")
    
    # Example 6: Generate structured text
    print("Generating structured text...")
    structure = [
        {'type': 'title', 'content': 'My Structured Document'},
        {'type': 'paragraph'},
        {'type': 'list', 'items': ['First item', 'Second item', 'Third item']},
        {'type': 'quote'},
        {'type': 'paragraph'}
    ]
    structured_text = text_data.generate_structured_text(structure)
    print(structured_text)
    print("\n" + "-"*80 + "\n")
    
    # Example 7: Generate text with entities
    print("Generating text with entities...")
    text, entities = text_data.generate_text_with_entities()
    print("Text:")
    print(text)
    print("\nEntities:")
    for entity_type, positions in entities.items():
        print(f"{entity_type}: {positions}")
    print("\n" + "-"*80 + "\n")
    
    # Example 8: Generate a dataset
    print("Generating a text dataset...")
    dataset_path = os.path.join(output_dir, 'text_dataset.json')
    texts = text_data.generate_text_dataset(num_texts=5, output_file=dataset_path, output_format='json')
    print(f"Generated {len(texts)} texts and saved to {dataset_path}")
    
    # Export examples to files
    print("\nExporting examples to files...")
    
    # Save paragraph
    paragraph_path = os.path.join(output_dir, 'paragraph.txt')
    with open(paragraph_path, 'w') as f:
        f.write(paragraph)
    print(f"Saved paragraph to {paragraph_path}")
    
    # Save article
    article_path = os.path.join(output_dir, 'article.txt')
    with open(article_path, 'w') as f:
        f.write(article)
    print(f"Saved article to {article_path}")
    
    # Save conversation
    conversation_path = os.path.join(output_dir, 'conversation.txt')
    with open(conversation_path, 'w') as f:
        f.write(conversation)
    print(f"Saved conversation to {conversation_path}")
    
    # Save structured text
    structured_path = os.path.join(output_dir, 'structured.txt')
    with open(structured_path, 'w') as f:
        f.write(structured_text)
    print(f"Saved structured text to {structured_path}")
    
    print("\nText data generation examples completed successfully!")

if __name__ == "__main__":
    main()
