import csv
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

STOPWORDS = STOPWORDS.union(set(["tambien", "vosotros", "voy", "y", "ya", "yo"... #For spanish - include how many words you want]))

text = ''

with open('Your_File_Name.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        text += ' '.join(row) + ' '

print(text)
##print(type(STOPWORDS))

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 500 * mask.astype(int)


wc = WordCloud(
    background_color="white",
    stopwords=STOPWORDS,
    repeat=True,
    mask=mask
    )

wc.generate(text)

plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.show()
