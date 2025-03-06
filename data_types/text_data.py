"""
Module for generating synthetic text data.
"""

import random
import string
import re
import json
import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from faker import Faker

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules
from utils import data_generators
from config import config

fake = Faker()

def generate_random_word(min_length: int = 3, max_length: int = 10, language: str = 'en') -> str:
    """
    Generate a random word.
    
    Args:
        min_length (int): Minimum length of the word
        max_length (int): Maximum length of the word
        language (str): Language code
        
    Returns:
        str: Random word
    """
    global fake
    fake.seed_instance(random.randint(0, 10000))
    
    if language != 'en':
        try:
            fake = Faker(language)
        except:
            fake = Faker('en')
    
    # Get a random word from lorem text
    words = fake.words(nb=30)
    
    # Filter words by length
    filtered_words = [word for word in words if min_length <= len(word) <= max_length]
    
    if filtered_words:
        return random.choice(filtered_words)
    else:
        # Fallback: generate a random string
        length = random.randint(min_length, max_length)
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_sentence(min_words: int = 5, max_words: int = 15, language: str = 'en') -> str:
    """
    Generate a random sentence.
    
    Args:
        min_words (int): Minimum number of words
        max_words (int): Maximum number of words
        language (str): Language code
        
    Returns:
        str: Random sentence
    """
    global fake
    if language != 'en':
        try:
            fake = Faker(language)
        except:
            fake = Faker('en')
    
    num_words = random.randint(min_words, max_words)
    sentence = fake.sentence(nb_words=num_words)
    
    return sentence

def generate_paragraph(min_sentences: int = 3, max_sentences: int = 8, 
                      min_words_per_sentence: int = 5, max_words_per_sentence: int = 15,
                      language: str = 'en') -> str:
    """
    Generate a random paragraph.
    
    Args:
        min_sentences (int): Minimum number of sentences
        max_sentences (int): Maximum number of sentences
        min_words_per_sentence (int): Minimum number of words per sentence
        max_words_per_sentence (int): Maximum number of words per sentence
        language (str): Language code
        
    Returns:
        str: Random paragraph
    """
    global fake
    if language != 'en':
        try:
            fake = Faker(language)
        except:
            fake = Faker('en')
    
    num_sentences = random.randint(min_sentences, max_sentences)
    sentences = []
    
    for _ in range(num_sentences):
        num_words = random.randint(min_words_per_sentence, max_words_per_sentence)
        sentences.append(fake.sentence(nb_words=num_words))
    
    paragraph = ' '.join(sentences)
    
    return paragraph

def generate_article(min_paragraphs: int = 3, max_paragraphs: int = 10,
                    min_sentences_per_paragraph: int = 3, max_sentences_per_paragraph: int = 8,
                    min_words_per_sentence: int = 5, max_words_per_sentence: int = 15,
                    include_title: bool = True, language: str = 'en') -> str:
    """
    Generate a random article.
    
    Args:
        min_paragraphs (int): Minimum number of paragraphs
        max_paragraphs (int): Maximum number of paragraphs
        min_sentences_per_paragraph (int): Minimum number of sentences per paragraph
        max_sentences_per_paragraph (int): Maximum number of sentences per paragraph
        min_words_per_sentence (int): Minimum number of words per sentence
        max_words_per_sentence (int): Maximum number of words per sentence
        include_title (bool): Whether to include a title
        language (str): Language code
        
    Returns:
        str: Random article
    """
    global fake
    if language != 'en':
        try:
            fake = Faker(language)
        except:
            fake = Faker('en')
    
    num_paragraphs = random.randint(min_paragraphs, max_paragraphs)
    paragraphs = []
    
    # Generate title if requested
    if include_title:
        title = fake.sentence(nb_words=random.randint(3, 8)).strip('.')
        paragraphs.append(f"# {title}\n")
    
    # Generate paragraphs
    for _ in range(num_paragraphs):
        num_sentences = random.randint(min_sentences_per_paragraph, max_sentences_per_paragraph)
        sentences = []
        
        for _ in range(num_sentences):
            num_words = random.randint(min_words_per_sentence, max_words_per_sentence)
            sentences.append(fake.sentence(nb_words=num_words))
        
        paragraph = ' '.join(sentences)
        paragraphs.append(paragraph)
    
    article = '\n\n'.join(paragraphs)
    
    return article

def generate_conversation(num_exchanges: int = 5, speakers: List[str] = None,
                         min_words_per_message: int = 5, max_words_per_message: int = 20,
                         language: str = 'en') -> str:
    """
    Generate a random conversation.
    
    Args:
        num_exchanges (int): Number of message exchanges
        speakers (list, optional): List of speaker names
        min_words_per_message (int): Minimum number of words per message
        max_words_per_message (int): Maximum number of words per message
        language (str): Language code
        
    Returns:
        str: Random conversation
    """
    global fake
    if language != 'en':
        try:
            fake = Faker(language)
        except:
            fake = Faker('en')
    
    # Generate speaker names if not provided
    if speakers is None or len(speakers) < 2:
        speakers = [fake.first_name() for _ in range(2)]
    
    conversation = []
    
    for i in range(num_exchanges * len(speakers)):
        speaker = speakers[i % len(speakers)]
        num_words = random.randint(min_words_per_message, max_words_per_message)
        message = fake.sentence(nb_words=num_words)
        conversation.append(f"{speaker}: {message}")
    
    return '\n'.join(conversation)

def generate_structured_text(structure: List[Dict[str, Any]], language: str = 'en') -> str:
    """
    Generate structured text based on a template.
    
    Args:
        structure (list): List of dictionaries defining the structure
            Each dictionary should have:
            - type (str): Type of element ('title', 'paragraph', 'list', 'quote', etc.)
            - ... other parameters specific to the element type
        language (str): Language code
        
    Returns:
        str: Generated structured text
    """
    global fake
    if language != 'en':
        try:
            fake = Faker(language)
        except:
            fake = Faker('en')
    
    elements = []
    
    for element in structure:
        element_type = element.get('type', 'paragraph')
        
        if element_type == 'title':
            level = element.get('level', 1)
            min_words = element.get('min_words', 3)
            max_words = element.get('max_words', 8)
            num_words = random.randint(min_words, max_words)
            title = fake.sentence(nb_words=num_words).strip('.')
            elements.append(f"{'#' * level} {title}")
        
        elif element_type == 'paragraph':
            min_sentences = element.get('min_sentences', 3)
            max_sentences = element.get('max_sentences', 8)
            min_words = element.get('min_words', 5)
            max_words = element.get('max_words', 15)
            paragraph = generate_paragraph(min_sentences, max_sentences, min_words, max_words, language)
            elements.append(paragraph)
        
        elif element_type == 'list':
            list_type = element.get('list_type', 'bullet')  # 'bullet' or 'numbered'
            min_items = element.get('min_items', 3)
            max_items = element.get('max_items', 7)
            min_words = element.get('min_words', 3)
            max_words = element.get('max_words', 10)
            
            num_items = random.randint(min_items, max_items)
            items = []
            
            for i in range(num_items):
                num_words = random.randint(min_words, max_words)
                item = fake.sentence(nb_words=num_words)
                
                if list_type == 'bullet':
                    items.append(f"- {item}")
                else:  # numbered
                    items.append(f"{i+1}. {item}")
            
            elements.append('\n'.join(items))
        
        elif element_type == 'quote':
            min_words = element.get('min_words', 10)
            max_words = element.get('max_words', 30)
            num_words = random.randint(min_words, max_words)
            quote = fake.sentence(nb_words=num_words)
            elements.append(f"> {quote}")
        
        elif element_type == 'code':
            language_name = element.get('language_name', 'python')
            min_lines = element.get('min_lines', 3)
            max_lines = element.get('max_lines', 10)
            
            num_lines = random.randint(min_lines, max_lines)
            
            if language_name == 'python':
                code_lines = [
                    "def example_function():",
                    "    # This is a sample function",
                    "    result = 0",
                ]
                
                for i in range(num_lines - 3):
                    code_lines.append(f"    result += {i+1}  # Incrementing result")
                
                code_lines.append("    return result")
                
            elif language_name == 'javascript':
                code_lines = [
                    "function exampleFunction() {",
                    "    // This is a sample function",
                    "    let result = 0;",
                ]
                
                for i in range(num_lines - 4):
                    code_lines.append(f"    result += {i+1};  // Incrementing result")
                
                code_lines.append("    return result;")
                code_lines.append("}")
                
            else:
                # Generic code
                code_lines = [f"Line {i+1} of code" for i in range(num_lines)]
            
            code_block = '\n'.join(code_lines)
            elements.append(f"```{language_name}\n{code_block}\n```")
        
        elif element_type == 'table':
            min_rows = element.get('min_rows', 3)
            max_rows = element.get('max_rows', 6)
            min_cols = element.get('min_cols', 2)
            max_cols = element.get('max_cols', 5)
            
            num_rows = random.randint(min_rows, max_rows)
            num_cols = random.randint(min_cols, max_cols)
            
            # Generate header
            header = []
            for i in range(num_cols):
                header.append(f"Column {i+1}")
            
            # Generate rows
            rows = []
            rows.append('| ' + ' | '.join(header) + ' |')
            rows.append('| ' + ' | '.join(['---'] * num_cols) + ' |')
            
            for i in range(num_rows):
                row = []
                for j in range(num_cols):
                    row.append(f"Cell {i+1},{j+1}")
                rows.append('| ' + ' | '.join(row) + ' |')
            
            elements.append('\n'.join(rows))
    
    return '\n\n'.join(elements)

def generate_text_with_entities(text_type: str = 'paragraph', entities: Dict[str, List[str]] = None,
                              min_entity_count: int = 3, max_entity_count: int = 10,
                              language: str = 'en', **kwargs) -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """
    Generate text with specific entities (names, locations, etc.) and their positions.
    
    Args:
        text_type (str): Type of text to generate ('paragraph', 'article', etc.)
        entities (dict, optional): Dictionary of entity types and possible values
        min_entity_count (int): Minimum number of entities to include
        max_entity_count (int): Maximum number of entities to include
        language (str): Language code
        **kwargs: Additional arguments for the text generation function
        
    Returns:
        tuple: (generated_text, entity_positions)
            - generated_text (str): The generated text
            - entity_positions (dict): Dictionary mapping entity types to lists of (start, end, value) tuples
    """
    global fake
    if language != 'en':
        try:
            fake = Faker(language)
        except:
            fake = Faker('en')
    
    # Default entities if not provided
    if entities is None:
        entities = {
            'PERSON': [fake.name() for _ in range(5)],
            'LOCATION': [fake.city() for _ in range(5)],
            'ORGANIZATION': [fake.company() for _ in range(5)],
            'DATE': [fake.date() for _ in range(5)]
        }
    
    # Generate base text
    if text_type == 'paragraph':
        base_text = generate_paragraph(language=language, **kwargs)
    elif text_type == 'article':
        base_text = generate_article(language=language, **kwargs)
    elif text_type == 'sentence':
        base_text = generate_sentence(language=language, **kwargs)
    else:
        base_text = generate_paragraph(language=language, **kwargs)
    
    # Split text into words
    words = re.findall(r'\b\w+\b', base_text)
    
    # Determine number of entities to insert
    num_entities = random.randint(min_entity_count, min(max_entity_count, len(words) // 5))
    
    # Choose random positions to insert entities
    positions = random.sample(range(len(words)), num_entities)
    positions.sort()  # Sort to process from start to end
    
    # Keep track of entity positions
    entity_positions = {entity_type: [] for entity_type in entities}
    
    # Insert entities
    offset = 0
    modified_text = base_text
    
    for pos in positions:
        # Choose a random entity type
        entity_type = random.choice(list(entities.keys()))
        # Choose a random entity value
        entity_value = random.choice(entities[entity_type])
        
        # Find the position of the word in the text
        word_to_replace = words[pos]
        pattern = r'\b' + re.escape(word_to_replace) + r'\b'
        
        # Find the first occurrence of the word after the current offset
        match = re.search(pattern, modified_text[offset:])
        if match:
            start_pos = offset + match.start()
            end_pos = offset + match.end()
            
            # Replace the word with the entity
            before = modified_text[:start_pos]
            after = modified_text[end_pos:]
            modified_text = before + entity_value + after
            
            # Update offset for next search
            offset = start_pos + len(entity_value)
            
            # Record entity position
            entity_positions[entity_type].append((start_pos, start_pos + len(entity_value), entity_value))
    
    return modified_text, entity_positions

def generate_text_dataset(num_texts: int = 10, text_type: str = 'paragraph',
                         output_file: str = None, output_format: str = 'txt',
                         language: str = 'en', **kwargs) -> List[str]:
    """
    Generate a dataset of synthetic texts.
    
    Args:
        num_texts (int): Number of texts to generate
        text_type (str): Type of texts to generate ('paragraph', 'article', 'sentence', 'conversation')
        output_file (str, optional): Path to save the dataset
        output_format (str): Format to save the dataset ('txt', 'json', 'csv')
        language (str): Language code
        **kwargs: Additional arguments for specific text types
        
    Returns:
        list: List of generated texts
    """
    # Set random seed if specified in config
    random_seed = config.get('random_seed')
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        fake.seed_instance(random_seed)
    
    # Generate texts
    texts = []
    for _ in range(num_texts):
        if text_type == 'paragraph':
            text = generate_paragraph(language=language, **kwargs)
        elif text_type == 'article':
            text = generate_article(language=language, **kwargs)
        elif text_type == 'sentence':
            text = generate_sentence(language=language, **kwargs)
        elif text_type == 'conversation':
            text = generate_conversation(language=language, **kwargs)
        elif text_type == 'structured':
            structure = kwargs.get('structure', [
                {'type': 'title', 'level': 1},
                {'type': 'paragraph'},
                {'type': 'list', 'list_type': 'bullet'},
                {'type': 'paragraph'},
                {'type': 'quote'}
            ])
            text = generate_structured_text(structure, language=language)
        elif text_type == 'with_entities':
            text, _ = generate_text_with_entities(
                text_type=kwargs.get('base_text_type', 'paragraph'),
                language=language,
                **kwargs
            )
        else:
            # Default to paragraph
            text = generate_paragraph(language=language, **kwargs)
        
        texts.append(text)
    
    # Save dataset if output file is specified
    if output_file:
        import os
        import json
        import csv
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        if output_format == 'txt':
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, text in enumerate(texts):
                    f.write(f"=== Text {i+1} ===\n{text}\n\n")
        
        elif output_format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([{'id': i, 'text': text} for i, text in enumerate(texts)], f, indent=2)
        
        elif output_format == 'csv':
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'text'])
                for i, text in enumerate(texts):
                    writer.writerow([i, text])
    
    return texts
