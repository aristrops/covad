from ollama import Client
import json
import os
import re
import pandas as pd

from mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import TaskType, Split, LabelName

client = Client(host="http://localhost:6000")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
MODEL_NAME = "gemma3:12b"
categories = ["hazelnut", "screw", "carpet"]
dataset_path = "/mnt/disk1/borsattifr/datasets/mvtec"


def concepts_to_vectors(concept_json, concept_list):
        return [int(concept_json.get(concept, False)) for concept in concept_list]

def extract_json(text):
    # Remove code block markers if present
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw response:", text)
        return {}


for category in categories: 
        test_dataset = MVTecDataset(task = TaskType.SEGMENTATION, root = dataset_path, category = category, split = "test")
        test_dataset.load_dataset()
        anomalous_test_samples = test_dataset.samples[(test_dataset.samples.split == "test") & (test_dataset.samples.label_index == LabelName.ABNORMAL)]
        normal_test_samples = test_dataset.samples[(test_dataset.samples.split == "test") & (test_dataset.samples.label_index == LabelName.NORMAL)]

        normal_concepts = []
        anomalous_concepts = []

        for i in range(len(normal_test_samples)):
                sample = normal_test_samples.iloc[i]
                image_path = sample["image_path"]
                message = "You are an expert evaluating an industrial image to detect anomalies. The image has been classified as normal, so there is no visible damage or defect. Among the following list of attributes, choose which ones are present in the image you are seeing." \
                "The concepts to select are: crack, uniform_sufrace, scratch, hole, twisted_tip, damaged_tip, stain, foreign_fiber, weave_break" \
                "Output the result as a JSON object of this form: {crack: true, scratch: false, ...}. Output ONLY the JSON object, nothing else."

                response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": message, "images": [image_path]}])

                #parse JSON string
                try:
                        concept_json = extract_json(response["message"]["content"])
                except Exception as e:
                        print(f"Error parsing response for {image_path}: {e}")
                        concept_json = {}
                
                vector = concepts_to_vectors(concept_json, concept_list)
                normal_concepts.append(vector)
        
        normal_concepts_df = pd.DataFrame(normal_concepts, columns = [f"concept_{c}" for c in concept_list])
        normal_test_samples = pd.concat([normal_test_samples.reset_index(drop=True), normal_concepts_df], axis = 1)
        

        for i in range(2):
                sample = anomalous_test_samples.iloc[i]
                image_path = sample["image_path"]
                message = "You are an expert evaluating an industrial image to detect anomalies. The image has been classified as anomalous, which implies that it shows a visible defect, alteration or damage. Among the following list of attributes, choose which ones are present in the image you are seeing." \
                "The concepts to select are: crack, uniform_sufrace, scratch, hole, twisted_tip, damaged_tip, stain, foreign_fiber, weave_break" \
                "Output the result as a JSON object of this form: {crack: true, scratch: false, ...}. Output ONLY the JSON object, nothing else."

                response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": message, "images": [image_path]}])

                #parse JSON string
                try:
                        concept_json = extract_json(response["message"]["content"])
                except Exception as e:
                        print(f"Error parsing response for {image_path}: {e}")
                        concept_json = {}
                
                vector = concepts_to_vectors(concept_json, concept_list)
                anomalous_concepts.append(vector)

        anomalous_concepts_df = pd.DataFrame(anomalous_concepts, columns = [f"concept_{c}" for c in concept_list])
        anomalous_test_samples = pd.concat([anomalous_test_samples.reset_index(drop=True), anomalous_concepts_df], axis = 1)

        print(anomalous_test_samples.head())