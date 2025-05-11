from ollama import Client
import os

client = Client(host="http://localhost:6000")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
MODEL_NAME = "gemma3:12b"

message = "You are an expert evaluating an industrial image to detect anomalies. The image has been classified as normal, so there is no visible damage or defect. Among the following list of attributes, choose which ones are present in the image you are seeing." \
        "Output the result as a JSON object of this form: {crack: true, scratch: false, ...}. The concepts to select are: crack, uniform_sufrace, scratch, hole, twisted_tip, damaged_tip, stain, foreign_fiber, weave_break"

message_anomalous = "You are an expert evaluating an industrial image to detect anomalies. This image is classified as anomalous, so there is a part of it that contains a defect with respect to the standard. Please provide a description of the image that specifically highlights which defect is present."
message_normal = "You are an expert evaluating an industrial image to detect anomalies. This image is classified as normal, so there is no defect visible. Please provide a description of the image with simple, one- or two-words attributes."
image = "/home/arianna_stropeni/thesis/Thesis/trial_images/normal_hazelnut.png"

response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": message, "images": [image]}])

print(response["message"]["content"])