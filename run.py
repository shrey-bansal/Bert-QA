import time
import nltk

from bert import QA
model = QA('model')

doc = open("data.txt").read()
q = "Which department filed an antitrust lawsuit against Microsoft in 1998 ?"
s = ""
start_time = time.time()
words = nltk.word_tokenize(doc)
print("Words in the document are "+ str(len(words))+" words")
if(len(words)>512):
    answer = model.predict(doc,q,150)
else:
    answer = model.predict(doc,q,len(words))


if(len(answer)==0):
    s+="No"
else:
    s+="Yes, "
    for i in range(len(answer)):
        t = answer[i]
        s+= t['answer'] + ", "
# 1975
print(q)
print("Final answer: ",s)
print("Time taken in seconds: " , (time.time() - start_time))
