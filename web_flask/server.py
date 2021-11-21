from flask import Flask, render_template, request, jsonify
from connection import s3_connection, get_s3_resource
from s3_config import BUCKET_NAME

app = Flask(__name__)

s3 = s3_connection()
s3_resource =get_s3_resource()

# S3 특정 폴더 내에 있는 파일 리스트를 얻는 함수
def get_file_list_s3(bucket, prefix=""):

    my_bucket = s3_resource.Bucket(bucket)
    file_objs = my_bucket.objects.filter(Prefix=prefix).all()
    file_names = [file_obj.key for file_obj in file_objs]
    return file_names
print(get_file_list_s3(bucket=BUCKET_NAME, prefix="male/image"))

# 옷 고를 수 있게 보여주기
@app.route('/view_img', methods=['GET'])
def get_photos():
    image_urls = []
    location = s3.get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
    path = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/male/image'
    img_list = get_file_list_s3(bucket=BUCKET_NAME, prefix="male/image")   #bucket='prodev-iboba', prefix="model-image/", file_extension=None)
    print(img_list)
    for img in img_list:
        img = img.replace("male/image", "")
        image_urls.append(path + img)
    return render_template('img_test.html', image_file=image_urls)

@app.route('/')
def sdsd():
   return 'Hello~~'


if __name__ == '__main__':
    #서버 실행
   app.run(debug = True, port=8000)
