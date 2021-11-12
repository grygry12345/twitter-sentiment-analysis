from bs4 import BeautifulSoup
import requests, json

resp = requests.get("http://www.netlingo.com/acronyms.php")
soup = BeautifulSoup(resp.text, "html.parser")
slangdict = []
# key = ""
# value = ""

# Get abbrevatinons from website and save in directory
index = 0
for div in soup.findAll("div", attrs={"class": "list_box3"}):
    for li in div.findAll("li"):
        for a in li.findAll("a"):
            # key =a.text
            # value = ('abb' ,li.text.split(key)[1].lower().replace(' ', '_'))
            # slangdict[key]=value
            item = {}
            item["index"] = index
            item["word"] = a.text
            item["meaning"] = li.text.split(a.text)[1].lower().replace(" ", "_")
            item["type"] = "abbrevation"

            slangdict.append(item)

            index += 1

# Finnaly dump directory varible json format
with open("tweet_abbrevations.json", "w") as f:
    json.dump(slangdict, f, indent=2)
