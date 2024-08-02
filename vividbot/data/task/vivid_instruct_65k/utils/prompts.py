from typing import Literal, Union

import numpy as np

DESCRIBE_VIDEO_PROMPTS = [
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


DESCRIBE_VIDEO_PROMPTS_VI = [
  """Hãy mô tả nội dung video một cách sinh động.""",
  """Viết một bản mô tả toàn diện về video, nắm bắt được điểm chính và những khoảnh khắc đáng nhớ.""",
  """Phân tích kỹ lưỡng video, bao gồm chủ đề chính và các yếu tố nổi bật.""",
  """Diễn đạt chính xác nội dung video, tập trung vào cốt truyện và hình ảnh quan trọng.""",
  """Trình bày phân tích chi tiết về những thành phần chủ yếu của video.""",
  """Mô tả cụ thể các hành động, nhân vật và bối cảnh xuất hiện trong video.""",
  """Trình bày chi tiết về khung cảnh, nhân vật và sự kiện diễn ra trong video.""",
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

GENERATE_QA_FROM_IMAGE_PROMPTS = {
  "reasoning": """Infer the content of the image and generate 1 to 3 pairs of questions focusing on mathematical reasoning, logical reasoning, causal reasoning, visual reasoning abilities and more, and answers in JSON format based on the image in Vietnamese language.
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
- Question: How many birds are the in the image and what happen if one bird flies away?
- Answer: There are three birds in the image. If one bird flies away, there will be two birds left in the image. The number of birds will decrease by one if one bird flies away.
5.
- Question: What could be the result of the question in the image?
- Answer: Based on the image, the question is to calculate the result of the equation 4 * 5 + 100. We can compute the final result by first multiplying 4 by 5, which equals 20. Then, we add 100 to the result, which gives us a final answer of 120. Therefore, the result of the equation 4 * 5 + 100 is 120.
And more questions that can be asked about the image content (what, where, when, why, how, etc.) with varying levels of complexity and logic. Harder questions that require more logical thinking are encouraged.
The answer should be descriptive, informative and correct in the form of complete sentences or complete phrases with proper grammar and punctuation.
Only return the list of pair of questions and answers in the following JSON list format without any additional information:
[{"question":"Q1","answer":"A1"},{"question":"Q2","answer":"A2"},...]""",
  "conversation": """Infer the content of the image and generate 5 to 10 pairs of CONVERSATIONAL questions and answers in JSON format based on the image in Vietnamese language.
The question should be relevant to the image content and the answer should be correct.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Examples of questions (do not need to follow the same as these are just examples, you must generate your own questions based on the image content):
1.
- Question: What are the colors of the bus in the image?
- Answer: The bus in the image is white and red.
2.
- Question: Where is the cat positioned in the image?
- Answer: The cat is positioned on top of the back of the couch in the living room.
3.
- Question: How many people are in the image?
- Answer: There are two people in the image, both on skis in the snow.
4.
- Question: Is the dog walking alone or with a person in the image?
- Answer: The dog is walking alone on the beach, with no person visible in the image.
5.
- Question: What is the physical appearance of the surfer?
- Answer: The surfer appears to be a muscular man in good physical shape.
And more questions that can be asked about the image content (what, where, when, why, how, etc.) with varying levels of complexity and logic. Harder questions that require more logical thinking are encouraged.
The answer should be descriptive, informative and correct in the form of complete sentences or complete phrases with proper grammar and punctuation.
Only return the list of pair of questions and answers in the following JSON list format without any additional information:
[{"question":"Q1","answer":"A1"},{"question":"Q2","answer":"A2"},...]""",
  "detail": """Infer the content of the image and generate 1 pair of questions and answers focusing on the DETAILS of the image in JSON format based on the image in Vietnamese language.
The question should be something to ask about the image content.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Examples of questions (do not need to follow the same as these are just examples, you must generate your own questions based on the image content):
1.
- Question: What do you see happening in this image?
- Answer: The scene depicts a lively plaza area with several people walking and enjoying their time. A man is standing in the plaza with his legs crossed, holding a kite in his hand. The kite has multiple sections attached to it, spread out in various directions as if ready for flight.\n\nNumerous people are scattered throughout the plaza, walking and interacting with others. Some of these individuals are carrying handbags, and others have backpacks. The image captures the casual, social atmosphere of a bustling plaza on a nice day.
2.
- Question: Can you describe the main features of this image for me?
- Answer: The image shows a lively beach scene where various beach-goers are enjoying a sunny day. People, some wearing bathing suits, are sitting and relaxing on the beach, working on their tans. There are couples and individuals scattered throughout the scene, relaxing on beach towels.\n\nMultiple beach umbrellas can be observed providing shade for some of the people, while a frisbee is visible, suggesting fun and games are taking place. A person can be seen reading a book, indicating that they are taking advantage of the relaxing atmosphere.\n\nThere are also a few handbags placed on the beach, likely holding beach-goers' belongings. The overall scene depicts a typical, enjoyable day at the beach for everyone involved.
3.
- Question: Explain the visual content of the image in great detail.
- This image captures several colorful kites flying in a field near a picturesque bay with a small lake. The kites come in a variety of shapes and sizes, with some featuring red, white, and blue streamers trailing from them. \n\nBy the scenic waterfront, there are several boats of different sizes docked or floating on the water. A group of people can be seen enjoying the outdoor atmosphere, walking along the river with their dogs. Some individuals have also parked their bicycles nearby, perhaps taking a break from a leisurely ride to appreciate the kites and the surrounding natural beauty. \n\nA bench located close to the edge of the field offers a perfect seating spot for onlookers, while others interact with each other, creating a peaceful and communal atmosphere in the area.
4.
- Question: Write a detailed description of the given image.
- Answer: The image shows a man standing on a street corner with a cart, selling unique umbrella-style hats. There are colorful umbrellas attached to poles sticking out of a basket next to the man, drawing the attention of passersby. Five umbrellas of varying sizes can be seen displayed, with one large umbrella placed in the foreground.\n\nBehind the man, two cars are parked along the side of the street, and another person is visible walking near the center of the scene. The man selling umbrella-style hats appears to be attracting interest or potentially waiting for customers to approach his makeshift street shop.
And more questions that can be asked about the image content (what, where, when, why, how, etc.) with varying levels of complexity and logic. Harder questions that require more logical thinking are encouraged.
The answer should be descriptive, informative and correct in the form of complete sentences or complete phrases with proper grammar and punctuation.
Only return the list of pair of questions and answers in the following JSON list format without any additional information:
[{"question":"Q1","answer":"A1"},{"question":"Q2","answer":"A2"},...]""",
}


def get_describe_video_prompt():
  # pick a random prompt
  return np.random.choice(DESCRIBE_VIDEO_PROMPTS)


def get_describe_video_prompt_vi():
  # pick a random prompt
  return np.random.choice(DESCRIBE_VIDEO_PROMPTS_VI)


def get_generate_qa_prompt():
  # pick a random prompt
  return np.random.choice(GENERATE_QA_PROMPTS)


def get_generate_qa_from_image_prompt(
  type: Union[
    Literal["conversation"], Literal["reasoning"], Literal["detail"]
  ] = "conversation",
):
  return GENERATE_QA_FROM_IMAGE_PROMPTS.get(type)
