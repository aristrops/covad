from ollama import Client
import os

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.patchcore.feature_compressor import CustomFeatureCompressor
from moviad.utilities.configurations import TaskType, Split, LabelName

client = Client(host="http://localhost:6000")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
MODEL_NAME = "gemma3:12b"
category = "carpet"

image_path = "/home/arianna_stropeni/thesis/Thesis/trial_images/normal_carpet.png"

screw_defects = ["thread_top", "thread_side", "scratch_neck", "scratch_head", "manipulated_tip"]
carpet_defects = ["thread", "metal_contamination", "hole", "cut", "color_contamination"]

anomalous_message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}. The image has been classified as anomalous, so there is a part of it that contains a defect with respect to the standard."\
f"The defect is metal contamination."\
f"Knowing these facts, please focus on its area and first provide a general description of it, and then from the description extract the most meaningful concepts."\
"The concepts should be defined in relationship to the image, in such a way that, observing the image, it is possible to clearly answer with yes or no about the presence of such a concept"\
"You can output five concepts or less, concepts can have more than one word, if this adds information, and should only be referred to visible features, avoiding speculations and assumptions"
#f"The possible defects that can be present are the following: {carpet_defects}. Please, select the two defects that are more likely to appear in the image, ordering them according to the probability that they are there."

normal_message = f"You are an expert evaluating an industrial image to detect anomalies. I provide an image of a {category}. The image has been classified as normal from an industrial point of view, so there is no visible defect, anomaly or issue."\
f"Knowing this fact, please provide a general description of the image, providing a characterization of what is visible, for example information about the texure, the color, any relevant features, ..."\
f"Then, from the description, extract the most meaningful concepts. The concepts should be defined in relationship to the image, in such a way that, observing the image, it is possible to clearly answer with yes or no about the presence of such a concept"\
"You can output five concepts or less, concepts can have more than one word, if this adds information, and should only be referred to visible features, avoiding speculations and assumptions"

first_response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": normal_message, "images": [image_path]}])

print(first_response["message"]["content"])

# defect = first_response["message"]["content"]

# follow_up_message = f"You are an expert evaluating an industrial image to detect anomalies. The image of a {category} I provide has been classified as anomalous, and the defect that is present in it is {defect}." \
# "Knowing this defect, please focus on its area and first provide a general description of it, and then from the description extract the most meaningful concepts"\
# "You can output five concepts or less, concepts can have more than one word, if this adds information, and should only be referred to visible features, avoiding speculations and assumptions"

# second_response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": follow_up_message, "images": [image_path]}])

# print(second_response["message"]["content"])