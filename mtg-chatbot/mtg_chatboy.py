import json
import requests

result = requests.get("https://api.scryfall.com/cards/search?q=name%3Adivination")

cards_string = result.content
cards = json.loads(cards_string)
print(cards["total_cards"])

for card in cards["data"]:
    print(card["name"])
