import numpy as np

DESCRIBE_VIDEO_PROMPTS = [
  #   """Describe only the visual content of the video without using the audio or transcript so that a normal person without visibility can interpret what happens in the video.
  # Don't use the audio or transcript of the video to describe the video content. Use only the visual content and chain the logical sequence of events in the video. Things described should show the correlation among them if they are related.
  # The description should first introduce the general overview of the video. Then describe the main objects, actions, and interactions in the video through time.
  # Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
  # Remember to use Vietnamese language to describe the video.""",
  """Paint a vivid picture of the video's content through a descriptive explanation in Vietnamese language.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.""",
  """Write a complete and exhaustive depiction of the video, capturing its essence and key moments in Vietnamese language.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.""",
  """Conduct a comprehensive and detailed examination of the video, analyzing its themes and elements in Vietnamese language.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.""",
  """Articulate the contents of the video with precision, emphasizing its storyline and visuals in Vietnamese language.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.""",
  """Present a detailed breakdown of the video's components, focusing on its essential parts in Vietnamese language.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.""",
  """Explore the video closely and provide a detailed account of its actions, characters, and setting in Vietnamese language.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.""",
  """Delve into the details of the video, including its setting, characters, and events in Vietnamese language.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.""",
]

GENERATE_QA_PROMPTS = [
  """Generate 2 different pairs of questions and answers in JSON format based on the description of the video (in which the description is generated for person without vision can understand the video content).
The questions should be relevant to the video content and the answers should be correct.
Diversify the types of questions and answers as much as possible.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to generate the questions and answers.
Examples of questions (do not need to follow the order and these are just examples, you must generate your own questions based on the video content):
- How does the guy create the multicolored vases using the glassblowing technique?
- What factors contribute to the collapse of the bridge during flooding?
- How did the Colorado Center for the blind interact with the interactive art wall mural?
- What does the inmate in blue apologize for?
- What does the man in the white shirt do with the bloody mass to keep the creatures away from himself?
- What led to the girl falling and landing on her neck while attempting a calisthenic trick on the horizontal bar?
- Can you describe what is happening in the video?
- Can you describe in detail what the little girl is doing in the video?
- ...
And more questions that can be asked about the video content (what, where, when, why, how, etc.) with varying levels of complexity and logic. Harder questions that require more logical thinking are encouraged.
The answer should be descriptive, informative and correct in the form of complete sentences or complete phrases with proper grammar and punctuation.
Sometimes, the description may repeat one information multiple times at the last part due to errors, you should avoid generating questions about that repeated information.
Only return the list of pair of questions and answers in the following JSON list format without any additional information:
[{{"question":"Q1","answer":"A1"}},{{"question":"Q2","answer":"A2"}},...]""",
  """Generate 2 different pairs of questions and answers in JSON format based on the description of the video (in which the description is generated for person without vision can understand the video content).
The questions should be relevant to the video content and the answers should be correct.
Diversify the types of questions and answers as much as possible.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to generate the questions and answers.
Examples of questions (do not need to follow the order and these are just examples, you must generate your own questions based on the video content):
- Did the lady in the black tanktop open the window?
- What was the lady in black with the gun trying to do?
- Is the young man in yellow sure that beat-boxing won't upset the boy with the black hood?
- Who is kneeling on the ground?
- What does the man in black feel about the blue creatures in the jars?
- ...
And more questions that can be asked about the video content (what, where, when, why, how, etc.) with varying levels of complexity and logic. Harder questions that require more logical thinking are encouraged.
The answer should be descriptive, informative and correct in the form of complete sentences or complete phrases with proper grammar and punctuation.
Sometimes, the description may repeat one information multiple times at the last part due to errors, you should avoid generating questions about that repeated information.
Only return the list of pair of questions and answers in the following JSON list format without any additional information:
[{{"question":"Q1","answer":"A1"}},{{"question":"Q2","answer":"A2"}},...]""",
  """Generate 5 to 10 different pairs of questions and answers in JSON format based on the description of the video (in which the description is generated for person without vision can understand the video content).
The questions should be relevant to the video content and the answers should be correct.
Diversify the types of questions and answers as much as possible.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to generate the questions and answers.
Examples of questions (do not need to follow the order and these are just examples, you must generate your own questions based on the video content):
- What's the video about?
- What are key points in the video?
- What is the color of the object X?
- How does the person in the video look?
- What is the position of the object in the video?
- How is the object X related to the object Y?
- Where is the man standing?
- What is the person in the video doing?
- Where is the mailbox located?
- What is the person's gender?
- What is the man in the blue shirt doing?
- ...
And more questions that can be asked about the video content (what, where, when, why, how, etc.) with varying levels of complexity and logic. Harder questions that require more logical thinking are encouraged.
The answer should be descriptive, informative and correct in the form of complete sentences or complete phrases with proper grammar and punctuation.
Sometimes, the description may repeat one information multiple times at the last part due to errors, you should avoid generating questions about that repeated information.
Only return the list of pair of questions and answers in the following JSON list format without any additional information:
[{{"question":"Q1","answer":"A1"}},{{"question":"Q2","answer":"A2"}},...]""",
]

GENERATE_QA_FROM_IMAGE_PROMPTS = [
  """Infer the content of the image and generate 1 to 3 pairs of questions and answers in JSON format based on the image in Vietnamese language.
The question should be relevant to the image content and the answer should be correct.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Examples of questions (do not need to follow the same as these are just examples, you must generate your own questions based on the image content):
1.
- Question: What precautions is the woman taking while walking in this weather?
- Answer: The woman is taking precautions while walking in the rainy weather by using an open umbrella to shield herself from the rain. The umbrella helps to keep her dry and protected from getting soaked by the rain, ensuring her comfort and safety while walking down the wet street. Holding the umbrella also indirectly indicates that the woman might be more attentive to her surroundings and cautious of potential hazards caused by the wet conditions, such as slippery surfaces, puddles, or splashing from passing vehicles.
2.
- Question: What might be the purpose of the vest the dog is wearing?
- The dog in the image is wearing a red vest, which could serve a few purposes. One possibility is that the vest offers warmth and protection for the dog in colder weather, especially if the dog has short fur or is sensitive to the cold. Another possibility is that the vest might be a specialized harness or vest designed for service, therapy, or working dogs. These vests are often used to help identify the dog's role and signal to others that the dog is on duty or in training. Finally, the vest could simply be a fashionable or decorative accessory that the dog's owner likes to dress their pet in for aesthetic reasons or to stand out in public.
3.
- Question: Why might someone enjoy pairing pizza with wine in this setting?
- In the image, a large piece of specialty thin-crust pizza with very little sauce, shaved meat, and goat cheese is served on a white plate next to a glass of wine. Pairing pizza with wine in this setting might be appealing for several reasons. Firstly, the combination of the unique flavors of the specialty pizza ingredients and the wine can create a pleasurable dining experience. It highlights the various tastes and textures of the pizza while allowing the wine to complement and enhance those flavors, creating a harmonious and satisfying meal. Additionally, the wine may elevate the casual nature of pizza, making the dining occasion feel more sophisticated and enjoyable. Lastly, the presentation of the meal, with a neatly set dining table, white plate, and wine glass, suggests an inviting atmosphere that allows diners to savor their meal in a relaxed and pleasurable setting.
4.
- Question: What is unusual about this situation when the man is tying the tie?
- Answer: The unusual aspect of this situation is that the man is attempting to tie a tie over a polo shirt. Typically, ties are worn with button-up shirts and are considered a more formal accessory. Polo shirts are generally more casual, and it is unconventional to pair them with a tie. This combination gives the scene an unusual and unconventional appearance.
5.
- Question: What potential risks are associated with the cat's proximity to electronic devices?
- Answer: With the cat lying next to mobile devices and other electronic equipment, there might be some potential risks such as knocking over the devices, causing damage to the devices, or accidentally activating buttons or sensitive touchscreens on the smartphones, resulting in unintended calls or messages. Additionally, if the cat were to chew on cables or cords associated with the electronic devices, it could pose a hazard to both the devices and the cat itself. In order to minimize these risks, it is advisable to keep smaller, fragile electronic devices away from pets or store them in places where pets cannot easily access them.
And more questions that can be asked about the image content (what, where, when, why, how, etc.) with varying levels of complexity and logic. Harder questions that require more logical thinking are encouraged.
The answer should be descriptive, informative and correct in the form of complete sentences or complete phrases with proper grammar and punctuation.
Only return the list of pair of questions and answers in the following JSON list format without any additional information:
[{{"question":"Q1","answer":"A1"}},{{"question":"Q2","answer":"A2"}},...]""",
  """Infer the content of the image and generate 3 to 10 pairs of questions and answers in JSON format based on the image in Vietnamese language.
The question should be relevant to the image content and the answer should be correct.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Examples of questions (do not need to follow the same as these are just examples, you must generate your own questions based on the image content):
1.
- Question: What are the colors of the bus in the image?
- Answer: The bus in the image is white and red.
2.
- Question: Where is the cat positioned in the image?
- The cat is positioned on top of the back of the couch in the living room.
3.
- Question: How many people are in the image?
- There are two people in the image, both on skis in the snow.
And more questions that can be asked about the image content (what, where, when, why, how, etc.) with varying levels of complexity and logic. Harder questions that require more logical thinking are encouraged.
The answer should be descriptive, informative and correct in the form of complete sentences or complete phrases with proper grammar and punctuation.
Only return the list of pair of questions and answers in the following JSON list format without any additional information:
[{{"question":"Q1","answer":"A1"}},{{"question":"Q2","answer":"A2"}},...]""",
]


def get_describe_video_prompt():
  # pick a random prompt
  return np.random.choice(DESCRIBE_VIDEO_PROMPTS)


def get_generate_qa_prompt():
  # pick a random prompt
  return np.random.choice(GENERATE_QA_PROMPTS)


def get_generate_qa_from_image_prompt():
  # pick a random prompt
  return np.random.choice(GENERATE_QA_FROM_IMAGE_PROMPTS)
