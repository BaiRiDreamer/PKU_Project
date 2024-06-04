import argparse
import os
from deepface import DeepFace
import json

def analyze_images(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            path = os.path.join(directory, filename)
            result = DeepFace.analyze(img_path=path, actions=['age', 'gender', 'race', 'emotion'])
            results[filename] = result
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze images in a directory.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing images')
    args = parser.parse_args()
    # Analyze images
    analysis_results = analyze_images(args.directory)
    # Save results to disk as JSON in the specified directory
    results_file_path = os.path.join(args.directory, 'analysis_results.json')
    with open(results_file_path, 'w') as file:
        json.dump(analysis_results, file, indent=4)

if __name__ == '__main__':
    main()
