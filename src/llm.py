import argparse
import os
import json
from openai import OpenAI
from init import analyze_images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

api_key = "sk-proj-9n8bP4r9IOy3Q5jG4dXMT3BlbkFJATEfXMoZQfLMtcIDXmMG"

client = OpenAI(
    api_key=api_key
)

def extract_confidence_score(response_text):
    match = re.search(r'\d+(\.\d+)?', response_text)
    if match:
        return float(match.group())
    else:
        raise ValueError(f"Could not interpret confidence score: {response_text}")


def display_images(directory, image_list):
    # Number of images
    num_images = len(image_list)
    
    # Create subplots with uniform width ratios
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for ax, (filename, confidence) in zip(axes, image_list):
        image_path = os.path.join(directory, filename)
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_title(f"{filename}\nConfidence: {confidence:.2f}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def load_analysis_results(directory):
    json_file = os.path.join(directory, 'analysis_results.json')
    if not os.path.exists(json_file):
        print("No analysis_results.json file found. Analyzing images...")
        analysis_results = analyze_images(directory)
        with open(json_file, 'w') as file:
            json.dump(analysis_results, file, indent=4)
        print("Image analysis complete and results saved.")
        return analysis_results
    else:
        with open(json_file, 'r') as file:
            return json.load(file)

def save_analysis_results(directory, results):
    json_file = os.path.join(directory, 'analysis_results.json')
    with open(json_file, 'w') as file:
        json.dump(results, file, indent=4)

def fetch_confidence_score(filename, prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    response = chat_completion.choices[0].message.content.strip()
    return filename, response

def match_images_with_description(analysis_results, user_description, k=3):
    matching_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for filename, details in analysis_results.items():
            details = details[0]
            description = f"Age: {details['age']}, Gender: {details['dominant_gender']}, " \
                          f"Race: {details['dominant_race']}, Emotions: {details['dominant_emotion']}."
            prompt = f"Does the following description match the user description '{user_description}'? " \
                     f"Description: {description}. PLEASE PROVIDE A CONFIDENCE SCORE FROM 0 TO 180."
            futures.append(executor.submit(fetch_confidence_score, filename, prompt))
        
        for future in as_completed(futures):
            filename, response = future.result()
            try:
                confidence = extract_confidence_score(response)
                matching_results.append((filename, confidence))
                
            except ValueError:
                print(f"Could not interpret confidence score for {filename}: {response}")

    matching_results.sort(key=lambda x: x[1], reverse=True)
    return matching_results[:k]
