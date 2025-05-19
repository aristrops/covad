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

    concepts = set()

    train_dataset = MVTecDataset(task = TaskType.SEGMENTATION, root = dataset_path, category = category, split = "train")
    train_dataset.load_dataset()

    test_dataset = MVTecDataset(task = TaskType.SEGMENTATION, root = dataset_path, category = category, split = "test")
    test_dataset.load_dataset()

    anomalous_samples = test_dataset.samples[test_dataset.samples.label_index == LabelName.ABNORMAL]
    print(f"Number of anomalous images: {len(anomalous_samples)}")

    normal_test_samples = test_dataset.samples[test_dataset.samples.label_index == LabelName.NORMAL]
    normal_samples = pd.concat([train_dataset.samples, normal_test_samples])
    print(f"Number of normal images: {len(normal_samples)}")

    for i in range(len(normal_samples)):
        sample = normal_samples.iloc[i]
        image_path = sample["image_path"]
        message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}. The image has been classified as normal from an industrial point of view, so there is no visible defect, anomaly or issue."\
                f"Knowing this fact, please provide a general description of the image, providing a characterization of what is visible, for example information about the texure, the color, any relevant features, ..."\
                f"Then, from the description, extract the most meaningful concepts. The concepts should be defined in relationship to the image, in such a way that, observing the image, it is possible to clearly answer with yes or no about the presence of such a concept"\
                "You can output five concepts or less, concepts can have more than one word, if this adds information, and should only be referred to visible features, avoiding speculations and assumptions."\
                "Please, your ONLY output should be the concepts, nothing else, written as a valid JSON array of strings."
        
        response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": message, "images": [image_path]}])

        try:
            concept_json = extract_json(response["message"]["content"])
        except Exception as e:
            print(f"Error parsing response for {image_path}: {e}")
            concept_json = []
        
        concepts.update(c.lower() for c in concept_json)
        
    
    for i in range(len(anomalous_samples)):
        sample = anomalous_samples.iloc[i]
        image_path = sample["image_path"]
        label = sample["label"]

        message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}. The image has been classified as anomalous, so there is a part of it that contains a defect with respect to the standard. The defect is {label}."\
                f"Knowing these facts, please focus on its area and first provide a general description of it, and then from the description extract the most meaningful concepts."\
                "The concepts should be defined in relationship to the image, in such a way that, observing the image, it is possible to clearly answer with yes or no about the presence of such a concept"\
                "You can output five concepts or less, concepts can have more than one word, if this adds information, and should only be referred to visible features, avoiding speculations and assumptions"\
                "Please, your ONLY output should be the concepts, nothing else, written as a valid JSON array of strings."
    
        response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": message, "images": [image_path]}])

        try:
            concept_json = extract_json(response["message"]["content"])
        except Exception as e:
            print(f"Error parsing response for {image_path}: {e}")
            concept_json = []
        
        concepts.update(c.lower() for c in concept_json)

    concepts = list(concepts)
    
    with open(f"{category}_concepts.json", "w") as f:
        json.dump(concepts, f)

    print(f"Extracted concepts for {category}:", concepts)