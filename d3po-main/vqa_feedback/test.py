from questions_generation import generate_qid_data_from_prompt


prompt = "A cat sitting on a red couch with a colorful pillow next to it."



questions = generate_qid_data_from_prompt(prompt)
print(questions)