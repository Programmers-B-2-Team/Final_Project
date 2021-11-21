from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, text
from werkzeug.utils import secure_filename
from sign_up import insert_generate, insert_user, get_user #insert_user_key
from connection import s3_connection
from s3_config import BUCKET_NAME
from make_user_key import get_user_key
app = Flask(__name__)

app.config.from_pyfile("config.py")
# 데이터 베이스와 연동해준다.
database = create_engine(app.config['DB_URL'], encoding = 'utf-8', max_overflow = 0)
app.database = database

s3 = s3_connection()

@app.route('/db')
def index():
   #new_user = request.json
   new_user = {
        'user_key'    : 'asdfszsdf2323asdxfxdfd',
        'model_image'   : 'c',
        'generate' : False
    }
   new_user_id = insert_user(new_user)
   new_user = get_user(new_user_id)
   return jsonify(new_user)

@app.route('/')
def sdsd():
   return 'Hello~~'

# 업로드 HTML 렌더링
@app.route('/upload')
def render_file():
   return render_template('upload.html')


# 파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # 이미지 객체
        f = request.files['file']
        


        ###########################
        ##### 이미지 전처리 #######
        ###########################

        # upload 시 user_key 생성 -> 파일 이름 규칙을 위해
        user_key = get_user_key(32)



        #저장할 경로 + 파일명

        # 처음 모델 업로드 시, 항상 상의: 0 , 하의: 0
        top_id, bottom_id = '00', '00'
        s3_path = 'model-image/' +  user_key + top_id + bottom_id

        s3.put_object(
            Bucket = BUCKET_NAME,
            Body = f,
            Key = s3_path,
            ContentType = f.content_type
        )
    location = s3.get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
    image_url = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/{s3_path}'

    new_user ={ 
        'model_url': image_url,
        'generate': False
        }

    insert_user(new_user)

    return render_template('view.html', image_url = image_url)
    #return 'uploads 디렉토리 -> 파일 업로드 성공!' + image_url



'''
# generate HTML 렌더링
# 상의 or 하의 item 선택 시 작동
@app.route('/generate')
def render_file():
    # 1. 이미지를 클릭할 시 해당 item 의 id 받아오기
    # image_url 과 item id 가 넘어옴
    # 1. 해당 url 이 user 테이블에 있는 지 조회
    # 2. 만약 있다면 바로 s3 버켓에서 해당 url 이미지 불러오기
    # 3. 없다면 딥러닝 모델을 거쳐 나온 아웃풋을 db에 저장하고 이를 불러오기
    model_url = model_url
    item_url = item_url
    item_num = get_item_num(item_url)
    if check_category(item_url) == 'top':
        new_model_url = model_url[:-4] + item_num + model_url[-2:]
    else:
        new_model_url = model_url[:-2] + item_num
    
    
    if check_model(new_model_url):
        return new_model_url
    else:
        item_url = item_url
        mask_url = get_mask_url()

        # model(item_url, mask_url, model_url)

        ####################
        ######모델추론######
        ####################
        # output_image = model.predict()

        s3_path = 'model-image/'+ new_model_url.split('/')[-1]
        s3.put_object(
            Bucket = BUCKET_NAME,
            Body = output_image,
            Key = s3_path,
            ContentType = output_image.content_type
        )
        location = s3.get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
        image_url = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/{s3_path}'
        gernerate = {
            'model_url' = model_url,
            'item_number' = item_num,
            'generate_url' = new_model_url
            }

        user = {
            'model_url' = new_model_url,
            'generate' = True
            }

        insert_generate(generate)
        
        insert_user(user)

        return generate_url
'''




if __name__ == '__main__':
    #서버 실행
   app.run(debug = True, port=8080)
