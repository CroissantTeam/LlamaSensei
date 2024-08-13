import json
try: 
    with open("stanford_cs229_l1.json","r",encoding="utf-8") as file:
        data = json.load(file)
except FileNotFoundError:
    print("The file was not found")
except json.JSONDecodeError:
    print("The file does not contain valid JSON")
except Exception as e:
    print(f"An error occurred: {str(e)}")

         #['metadata', 'results']
print(data["results"]["channels"][0]["alternatives"][0].keys())
                    #only      #only    #only       #only #['transcript', 'confidence', 'words', 'paragraphs']

# print(print(data["metadata"]))
metadata = data["metadata"]
#only consider paragraphs
# print(len(data["results"]["channels"][0]["alternatives"][0]["paragraphs"]["paragraphs"]))
                                                                            #['transcript', 'paragraphs']
paragraphs = data["results"]["channels"][0]["alternatives"][0]["paragraphs"]["paragraphs"]
# print(len(paragraphs))
# print("======",paragraphs[0].keys()) #['sentences', 'start', 'end', 'num_words']
# paragraph = paragraphs[0]
# sentence = paragraph["sentences"][1]
# print(sentence)



doc=[]
for paragraph in paragraphs:
    text = " ".join([sentence["text"] for sentence in paragraph["sentences"]])
    start = paragraph["start"]
    end = paragraph["end"]
    doc.append((text,start,end))

# def load_data(simple_output=True):
    
