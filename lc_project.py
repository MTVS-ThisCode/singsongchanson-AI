import openai
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_KEY")

TEMPLATE = """
Based on the depiction of '{depiction}', compose a '{genre}'-genre track where a series of captivating melodies evolve and 
transition into one another, creating a flowing narrative throughout the entire duration, blending the sounds associated 
with '{element}' and natural elements. Ensure the melodies remain the focus, ready to be paired with lyrics for singing, 
while presenting a structured musical narrative akin to mainstream pop, with clear progression through varying musical sections such as verses, 
choruses, and a bridge. The beats should provide a rhythmic foundation, with '{instrument}'-appropriate instruments enhancing the tranquility and 
liveliness associated with '{element}', intertwining with the evolving melodies to create a rich, 
dynamic soundscape with a coherent musical journey. Genre: '{genre}',
"""

def analyze_input(base_sentence):
    prompt = f"Given the depiction: '{base_sentence}', please provide the genre, element, and instrument in the following format:\n\nGenre: [genre]\nElement: [element]\nInstrument: [instrument]"

    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=500
    )
    response_text = response.choices[0].text.strip()
    
    genre = response_text.split("Genre: ")[1].split("\n")[0]
    element = response_text.split("Element: ")[1].split("\n")[0]
    instrument = response_text.split("Instrument: ")[1].split("\n")[0]

    return {
        'depiction': base_sentence,
        'genre': genre,
        'element': element,
        'instrument': instrument
    }

# 사용자로부터 문장 입력 받기
# base_sentence = input("문장을 입력하세요: ")
# analyzed_data = analyze_input(base_sentence)
# filled_template = TEMPLATE.format(**analyzed_data)
# print(filled_template)