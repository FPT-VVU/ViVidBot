import numpy as np

DESCRIBE_VIDEO_PROMPTS = [
  """Describe only the visual content of the video without using the audio or transcript so that a normal person without visibility can interpret what happens in the video.
Don't use the audio or transcript of the video to describe the video content. Use only the visual content and chain the logical sequence of events in the video. Things described should show the correlation among them if they are related.
The description should first introduce the general overview of the video. Then describe the main objects, actions, and interactions in the video through time.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to describe the video.""",
  """Paint a vivid picture of the video's content through a descriptive explanation.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to describe the video.""",
  """Write a complete and exhaustive depiction of the video, capturing its essence and key moments.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to describe the video.""",
  """Conduct a comprehensive and detailed examination of the video, analyzing its themes and elements.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to describe the video.""",
  """Articulate the contents of the video with precision, emphasizing its storyline and visuals.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to describe the video.""",
  """Present a detailed breakdown of the video's components, focusing on its essential parts.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to describe the video.""",
  """Explore the video closely and provide a detailed account of its actions, characters, and setting.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to describe the video.""",
  """Delve into the details of the video, including its setting, characters, and events.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to describe the video.""",
]

GENERATE_QA_PROMPTS = [
  """Generate 2 different pairs of questions and answers in JSON format based on the description of the video (in which the description is generated for person without vision can understand the video content).
The questions should be relevant to the video content and the answers should be correct.
Diversify the types of questions and answers as much as possible.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to generate the questions and answers.
Examples of questions ranging from normal conversations to causal reasoning or complex reasoning (do not need to follow the order and these are just examples, you must generate your own questions based on the video content):
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
  """Generate 5 to 20 different pairs of questions and answers in JSON format based on the description of the video (in which the description is generated for person without vision can understand the video content).
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


def get_describe_video_prompt():
  # pick a random prompt
  return np.random.choice(DESCRIBE_VIDEO_PROMPTS)


def get_generate_qa_prompt():
  # pick a random prompt
  return np.random.choice(GENERATE_QA_PROMPTS)
