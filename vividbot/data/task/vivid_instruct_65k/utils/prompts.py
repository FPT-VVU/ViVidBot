DESCRIBE_VIDEO_PROMPT = """Describe only the visual content of the video without using the audio or transcript so that a normal person without visibility can interpret what happens in the video.
Don't use the audio or transcript of the video to describe the video content. Use only the visual content and chain the logical sequence of events in the video. Things described should show the correlation among them if they are related.
The description should first introduce the general overview of the video. Then describe the main objects, actions, and interactions in the video through time.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to describe the video."""

GENERATE_QA_PROMPT = """Generate 5 different pairs of questions and answers in JSON format based on the description of the video (in which the description is generated for person without vision can understand the video content).
The questions should be relevant to the video content and the answers should be correct.
Diversify the types of questions and answers as much as possible.
Remember to avoid harmful, inappropriate, or offensive content in the questions and answers.
Remember to use Vietnamese language to generate the questions and answers.
Examples of questions (do not need to follow the order and these are just examples, you must generate your own questions based on the video content):
- What's the video about?
- What are key points in the video?
- What is the color of the object X?
- What is the person in the video holding and what are its characteristics?
- How does the person in the video look?
- What is the position of the object in the video?
- How is the object X related to the object Y?
- ...
And more questions that can be asked about the video content (what, where, when, why, how, etc.) with varying levels of complexity and logic. Harder questions that require more logical thinking are encouraged.
The answer should be informative and correct in the form of complete sentences or complete phrases with proper grammar and punctuation.
Sometimes, the description may repeat one information multiple times at the last part due to errors, you should avoid generating questions about that repeated information.
Only return the list of pair of questions and answers in the following JSON list format without any additional information:
[{{"question":"Q1","answer":"A1"}},{{"question":"Q2","answer":"A2"}},...]"""
