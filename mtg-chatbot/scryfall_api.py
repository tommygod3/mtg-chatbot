import json
import requests
from PIL import Image
from io import BytesIO

class Scryfall:
    def __init__(self):
        pass



    def get_card(self, name):
        encoded = name.replace(" ", "+")
        url = f"https://api.scryfall.com/cards/named?fuzzy={encoded}"
        result = requests.get(url)
        response_string = result.content
        card = json.loads(response_string)
        print(card["name"])



# result = requests.get("https://api.scryfall.com/cards/random")

# card_string = result.content
# card = json.loads(card_string)
# print(card)


# response = requests.get(card["image_uris"]["large"])
# img = Image.open(BytesIO(response.content))
# img.show()
