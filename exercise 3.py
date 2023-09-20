import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter

# 读取Moby Dick文本
moby_dick_text = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')

# Tokenization
tokens = word_tokenize(moby_dick_text.lower())

# Stop-words过滤
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# Parts-of-Speech (POS)标注
pos_tags = pos_tag(filtered_tokens)

# POS频率计算
pos_counts = Counter(tag for word, tag in pos_tags)
top_pos = pos_counts.most_common(5)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word, tag in pos_tags[:20]]

# 绘制POS频率分布柱状图
pos_labels, pos_values = zip(*pos_counts.items())
plt.bar(pos_labels, pos_values)
plt.xlabel('Parts of Speech')
plt.ylabel('Frequency')
plt.title('POS Frequency Distribution')
plt.show()

# 显示结果
print("Top 5 Parts of Speech and their counts:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

print("\nLemmatized Tokens:")
print(lemmatized_tokens)