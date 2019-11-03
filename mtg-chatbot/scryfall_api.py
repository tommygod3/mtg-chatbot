import json
import requests

class Scryfall:
    def __init__(self):
        pass

    def get_card(self, name):
        encoded = name.replace(" ", "+")
        url = f"https://api.scryfall.com/cards/named?exact={encoded}"

        response = requests.get(url)
        response_string = response.content
        card = json.loads(response_string)

        if response.status_code != requests.codes["✓"]:
            raise RuntimeError(card["details"])
        
        return card

    def random_card(self):
        response = requests.get("https://api.scryfall.com/cards/random")
        response_string = response.content
        card = json.loads(response_string)
        return card
