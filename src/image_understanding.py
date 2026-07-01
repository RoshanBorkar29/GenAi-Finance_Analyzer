# src/image_understanding.py
from pathlib import Path
import os
from PIL import Image
import google.genai as genai
from dotenv import load_dotenv

class ImageAnalyzer:

    def __init__(self, output_dir="data/extracted_text"):
        load_dotenv()
        # ✅ FIX: Capitalized 'Client'
        self.client = genai.Client()
        self.output_dir = Path(output_dir)
        self.create_output_folder()

    def create_output_folder(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_image(self, image_path, prompt):
        image = Image.open(image_path)
        # ✅ FIX: Properly structure the call to the model
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image]
        )
        return response.text

    def save_description(self, image_path, description):
        image_path = Path(image_path)
        output_file = self.output_dir / f"{image_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(description)
        return output_file

    def analyze_folder(self, image_paths, prompt):
        description_files = []
        for image_path in image_paths:
            description = self.analyze_image(image_path, prompt)
            file_path = self.save_description(image_path, description)
            description_files.append(file_path)
            
        # ✅ FIX: Moved out of the loop so it processes all images
        return description_files