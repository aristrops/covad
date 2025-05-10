from ollama import Client
import json

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import TaskType, Split

dataset_path = "/mnt/disk1/borsattifr/mvtec"
category = "hazelnut"
client = Client(host="http://localhost:6000")
MODEL_NAME = "gemma3:12b"

dataset = MVTecDataset(task = TaskType.SEGMENTATION, root = dataset_path, category = category, split = "test")
dataset.load_dataset()
dataset_images = dataset.samples.sample(len(dataset))

concept_vectors = []
concept_list = ["crack", "scratch", "hole", "twisted_tip", "broken_tip", "stain"k, "unpicking", "cut", "slub", "color_fly_yarn"]

def concepts_to_vectors(concept_json, concept_list):
        return [int(concept_json.get(concept, False)) for concept in concept_list]

for _, sample in dataset_images.iterrows():
        image_path = sample["image_path"]
        message = "You are an expert evaluating an industrial image to detect anomalies. Among the following list of attributes, choose which ones are present in the image you are seeing." \
        "Output the result as a JSON object."

        response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": "message", "images": [image_path]}])

        #parse JSON string
        try:
                concept_json = json.loads(response["content"])
        except Exception as e:
                print(f"Error parsing response for {image_path}: {e}")
                concept_json = {}
        
        vector = concepts_to_vectors(concept_json, concept_list)
        concept_vectors.append(vector)