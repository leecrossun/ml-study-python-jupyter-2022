# rnn_tokenize
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

en_text = "A dog run back corner near spare bedrooms"
print(word_tokenize(en_text))

kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
