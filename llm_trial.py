from ollama import Client
import os

client = Client(host="http://localhost:6000")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
MODEL_NAME = "gemma3:12b"

message = "Analyze the following image and return a list of key visual concepts, objects, and themes found in it. Output the result as a JSON object."
image = "/home/arianna_stropeni/thesis/Thesis/broken_nut.png"

response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": message, "images": [image]}])

print(response["message"]["content"])