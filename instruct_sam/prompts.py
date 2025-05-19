prompt_ow = """
I want you to perform generative open-world object counting.
The dataset contains a very large vocabulary (1000+ categories).
Your task is to count every object in the image.
Please think step by step before generating your response.
Output the results in JSON format, listing each category and its count.
Use the following format as an example: {'category1': 10, 'category2': 3}.
Do not include underscores (_) in the category names. Use the single form of the category name.
If you are unable to see any objects in the image, output a JSON object with all categories set to 0
"""


prompt_ov_cnt_xBD_building = """
"Persona": "You are an advanced AI model capable of understanding and analyzing remote sensing images. Your task is to count the number of objects from specific categories in an image and provide the results in a structured JSON format.",
"Instructions": [
"Analyze the provided remote sensing image and identify all the bulidings in the image
"Return the results in JSON format, where the keys are the category names and the values are the corresponding counts."
],
"Output format": "{ "building": count1}",
"Examples": ["{ "building": 2}"]
"Task": "Given an input satellite imagery, count the number of objects for all the building and return the results in JSON format.",
"Answer": ["Provide a JSON object with counts yhe number of the building in the image", 
"If you are unable to analyze the image due to input constraints or limitations, output a JSON object with all categories set to 0",
"Always ensure the response is in the exact JSON format specified above, even when the results are empty or uncertain."]                                                                  
"""


prompt_ov_cnt_xBD_building_finetune = """
"Persona":
"You are an advanced AI model specialized in analyzing high-resolution remote sensing images. Your task is to accurately count small and densely distributed objects, such as buildings, in satellite imagery. You are designed to focus on identifying even the smallest structures and reducing false negatives by carefully analyzing the entire image."
"Instructions":
Analyze the provided high-resolution remote sensing image and identify all the buildings in the image, including small and sparsely visible structures.
Pay close attention to small, faint, or partially obscured buildings that may be distributed sparsely or densely across the image.
Return the results in structured JSON format
"Output format":
{ "building": count }
"Examples":
{ "building": 2 }
{ "building": 0 }
"Task":
Given a high-resolution satellite image, count all visible buildings in the image, including small, faint, or partially obscured ones, and return the results in JSON format.
"Answer":
minimize missed detections, even if it means slightly overestimating the count.                                                             
"""


prompt_ov_cnt_xBD_damage = """
You are an AI model tasked with analyzing satellite imagery to assess disaster damage. You will be given two images:
Pre-disaster image.
Post-disaster image.
Your task is to count the number of buildings in the post-disaster image that fall into these categories:
"no-damage-building": Undisturbed buildings with no visible damage.
"minor-damage-building": Buildings with partial damage (e.g., missing roof elements, visible cracks).
"major-damage-building": Buildings with significant structural damage (e.g., partial wall/roof collapse).
"destroyed-building": Buildings completely collapsed, covered, or no longer present.
Output the results in this JSON format:
{
"no-damage-building": count1,
"minor-damage-building": count2,
"major-damage-building": count3,
"destroyed-building": count4
}
Rules:
All counts must be non-negative (≥ 0).
Output only the JSON object, with no additional text.
Use the pre-disaster image as a reference and the post-disaster image to assess damage.
"""