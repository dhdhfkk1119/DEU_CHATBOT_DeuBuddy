from konlpy.tag import Komoran

komoran = Komoran(userdic='../utils/user_dic.tsv')
text = "오늘 날씨는 구름이 많아요"
pos = komoran.pos(text)
print(pos)