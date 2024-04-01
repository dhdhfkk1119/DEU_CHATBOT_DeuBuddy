from PIL import Image
from io import BytesIO
import glob
import os

# 현재 경로와 저장 경로 설정
path = os.path.dirname(os.path.realpath(__file__))
save_path = "/Users/chanmin/DEU_CHATBOT_deudeu/front_end/static/img/"
os.chdir(path)
print("path is : " + path)

# 배열 선언
image_list_png = []
image_list_jpg = []

# png 파일 불러오기 + jpg 변환
read_files_png = glob.glob('/Users/chanmin/DEU_CHATBOT_deudeu/front_end/static/img/*.png') # 절대 경로
# read_files_png.sort()
print(read_files_png)
print(len(read_files_png))

# .png 뺀 파일명 추출
image_list = os.listdir('/Users/chanmin/DEU_CHATBOT_deudeu/front_end/static/img')
print("image list : ", image_list)
print(len(image_list))
search = '.png'
for i, word in enumerate(image_list):
    if search in word:
        image_list_png.append(word.strip(search))
search = '.jpg'
print(image_list_png)
print(len(image_list_png))
print(image_list)
print(len(image_list))

# png -> jpg
cnt2 = 0
for f in read_files_png:
    img = Image.open(f).convert('RGB')
    img.save("/Users/chanmin/DEU_CHATBOT_deudeu/front_end/static/img/"+image_list_png[cnt2]+".jpg", 'jpeg')
    cnt2 += 1

# jpg 파일 resizing
read_files_jpg = glob.glob("/Users/chanmin/DEU_CHATBOT_deudeu/front_end/static/img/*.jpg")

# .jpg 뺀 파일명 추출
image_list = os.listdir("/Users/chanmin/DEU_CHATBOT_deudeu/front_end/static/img")
print(image_list)
print(len(image_list))
for i, word in enumerate(image_list):
    if search in word:
        image_list_jpg.append(word.strip(search))
print(image_list_jpg)
print(len(image_list_jpg))

cnt=0
for f in read_files_jpg:
    print(f)
    img = Image.open(f)
    img = img.resize((int(img.width / 2), int(img.height / 2))) # 이미지 크기 줄이기
    buffer = BytesIO()
    img.save(buffer, 'jpeg', quality=70)
    buffer.seek(0)
    with open(save_path + image_list_jpg[cnt] + '.jpg', 'wb') as nfile:
        nfile.write(buffer.getvalue())
    cnt += 1

# img 폴더의 모든 png 파일 삭제 -> 변환된 jpg 파일만 남김
[os.remove(f) for f in glob.glob(save_path+"*.png")]