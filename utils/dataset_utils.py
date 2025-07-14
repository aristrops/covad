import json
import re
import torch
from ollama import Client

client = Client(host="http://localhost:6000")

#parse JSON object
def extract_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw response:", text)
        return {}


#convert concepts to binary vectors
def concepts_to_vectors(concept_json, concept_list):
        return [int(concept_json.get(concept, False)) for concept in concept_list]


#first VLM query
def first_vlm_query(category, model_name, sample):
    image_path = sample["image_path"]
    label = sample["label"]
    
    if label == "good":
        message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}. The image has been classified as normal from an industrial point of view, so there is no visible defect, anomaly or issue."\
        f"Knowing this fact, please provide a general description of the image, providing a characterization of what is visible, for example information about the texure, the color, any relevant features, ..."\
        f"Then, from the description, extract the most meaningful concepts. The concepts should be defined in relationship to the image, in such a way that, observing the image, it is possible to clearly answer with yes or no about the presence of such a concept"\
        "Concepts should be referred to visible features only, avoiding speculations or assumptions."\
        "Avoid concepts that are diffused and cannot be grounded on an area of the image, e.g. AVOID concepts such as 'surface discontinuity', 'irregular shape'."\
        "You can output five concepts or less, concepts can have more than one word, if this adds information."\
        "Please, your ONLY output should be the concepts, nothing else, written as a valid JSON array of strings."

    else:
        message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}. The image has been classified as anomalous, so there is a part of it that contains a defect with respect to the standard. The defect is {label}."\
        f"Knowing these facts, please focus on its area and first provide a general description of it, and then from the description extract the most meaningful concepts."\
        "The concepts should be defined in relationship to the image, in such a way that, observing the image, it is possible to clearly answer with yes or no about the presence of such a concept"\
        "Concepts should be referred to visible features only, avoiding speculations or assumptions."\
        "Avoid concepts that are diffused and cannot be grounded on an area of the image, e.g. AVOID concepts such as 'surface discontinuity', 'irregular shape'."\
        "You can output five concepts or less, concepts can have more than one word, if this adds information."\
        "Please, your ONLY output should be the concepts, nothing else, written as a valid JSON array of strings."

    
    response = client.chat(model=model_name, messages=[{"role": "user", "content": message, "images": [image_path]}])

    try:
        concept_json = extract_json(response["message"]["content"])
    except Exception as e:
        print(f"Error parsing response for {image_path}: {e}")
        concept_json = []
    
    return concept_json


#second VLM query
def second_vlm_query(category, model_name, sample, concept_list):
    image_path = sample["image_path"]
    label = sample["label"]
    if label != "good":
        mask_path = sample["mask_path"]

    if label == "good":
        message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}"\
                "The image has been classified as normal, which implies that there is no visible defect, anomaly or issue."\
                f"Knowing this, choose which concepts you see in the image among the following list of attributes: {concept_list}."\
                "Output the result as a JSON object of this form: {concept_1: true, concept_2: false, ...}. Output ONLY the JSON object, nothing else."
        
        response = client.chat(model=model_name, messages=[{"role": "user", "content": message, "images": [image_path]}])

    else:
        message = f"You are an expert evaluating an industrial image to detect anomalies. The first image I provide is an image of a {category}"\
                f"The image has been classified as anomalous, which implies that it shows a visible defect, alteration or damage. The defect is {label}, and the second image shows the localization of the defect with an anomaly mask."\
                f"Knowing this, choose which concepts you see in the image among the following list of attributes: {concept_list}."\
                "Output the result as a JSON object of this form: {concept_1: true, concept_2: false, ...}. Output ONLY the JSON object, nothing else."

        response = client.chat(model=model_name, messages=[{"role": "user", "content": message, "images": [image_path, mask_path]}])

    try:
        concept_json = extract_json(response["message"]["content"])
    except Exception as e:
            print(f"Error parsing response for {image_path}: {e}")
            concept_json = {}
    
    vector = concepts_to_vectors(concept_json, concept_list)

    return vector


