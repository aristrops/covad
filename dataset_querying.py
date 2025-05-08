from ollama import Client

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import TaskType, Split

dataset_path = "/mnt/disk1/borsattifr/mvtec"
category = "hazelnut"
client = Client(host="http://localhost:6000")
MODEL_NAME = "gemma3:12b"

dataset = MVTecDataset(task = TaskType.SEGMENTATION, root = dataset_path, category = category, split = "test")
dataset.load_dataset()

for image in dataset: 

        

        response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": "tell me a joke"}])

        print(response["message"]["content"])