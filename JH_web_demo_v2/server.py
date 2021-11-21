from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, text
from werkzeug.utils import secure_filename
from sign_up import get_items, get_user, insert_generate, insert_user, get_item_num, check_category, check_model, get_mask_url
from connection import s3_connection, get_s3_resource
from s3_config import BUCKET_NAME
from make_user_key import get_user_key
from preprocess import resizing_human
from create_model import *
import io
from inference.inference import inference_image
import os
from flask_share import Share


app = Flask(__name__)
share = Share(app)


app.config.from_pyfile("config.py")

# 데이터 베이스와 연동해준다.
database = create_engine(app.config['DB_URL'], encoding='utf-8', max_overflow=0)
app.database = database

# s3 storage 관련 변수
s3 = s3_connection()
s3_resource = get_s3_resource()
location = s3.get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
path = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/'

# 전처리 모델을 사전에 불러오는 코드
net = create_model('Unet_human')
net.eval()

# 대문 페이지
@app.route('/')
def sss():
    return render_template('index.html')

# 테스트용 페이지
'''
@app.route('/test')
def test():
    items= get_items("top","w")[0][2]
    print(items)
    return ""
'''

# landing_page에 이미지 보여주기
@app.route('/select', methods=['GET'])
def get_photos():
    default_model = get_user(1)['model_url']
    default_items= get_items("top","w")
    image_urls = []
    image_urls.append(default_model)
    for i in range(len(default_items)):
        image_urls.append(default_items[i][2])
    return render_template('select.html', image_file=image_urls)


# female / male 버튼을 누르면 모델 이미지 바꿔서 보여주기
@app.route('/click_gender', methods=['GET'])
def get_model_photo():
    parameter_dict = request.args.to_dict()
    gender = parameter_dict['gender']
    if gender == "w":
        model_image = get_user(1)['model_url']
    else:
        model_image = get_user(2)['model_url']

    # html의 조각만 return해줌
    return render_template('model_image.html', image_name=model_image)


# top / bottom 버튼을 누르면 젠더/ 카테고리에 맞는 아이템 이미지 바꿔주기
@app.route('/click_category', methods=['GET'])
def get_items_photo():
    parameter_dict = request.args.to_dict()
    category = parameter_dict['category']
    gender = parameter_dict['gender']
    items_list = get_items(category,gender)
    print(items_list)
    img_urls = []
    for i in range(len(items_list)):
        img_urls.append(items_list[i][2])
    # html의 조각만 return해줌
    return render_template('item_images.html', img_urls=img_urls)

# 기존 아이디이면 아이템을 눌렀을 때 모델 자리를 생성된 이미지로 바꿔주기
@app.route('/click_item2', methods=['GET', 'POST'])
def get_paired_photo():
    parameter_dict = request.args.to_dict()
    gender = parameter_dict['gender']
    print(gender)
    category = parameter_dict['category']
    top_id = parameter_dict['top_id']
    bottom_id = parameter_dict['bottom_id']

    img_path = "img/model/"+gender+top_id+bottom_id+".jpg"
    return render_template('model_image_static.html', image_name=img_path)



# 아이템을 눌렀을 때 모델 자리를 생성된 이미지로 바꿔주기
@app.route('/click_item', methods=['GET', 'POST'])
def get_generated_photo():
    parameter_dict = request.args.to_dict()
    gender = parameter_dict['gender']
    category = parameter_dict['category']
    item_id = parameter_dict['item_id']
    s3_url =parameter_dict['s3_url']    
    print()
    print('s3_url', s3_url)
    print('item_id', item_id)
    print()


    # 추론 함수에 들어갈 input image들 지정해주기
    key = s3_url.replace("https://project-dev-b2.s3.ap-northeast-2.amazonaws.com/","")
    name = key.replace("model_image/","")
    
    #user_key, top_id, bottom_id 받아오기
    top_id, bottom_id = name.split('.')[0][-2], name.split('.')[0][-1] 
    if category == 'top':
        top_id = item_id[-1]
    else:
        bottom_id = item_id[-1]
    print()
    print('top_id', top_id)
    print('bottom_id', bottom_id)
    print()
    user_key = name.split('.')[0][:-2]
    path='/home/jh/Final_Project/JH_web_demo_v2/inference/dataset/'
    s3_resource.Bucket(BUCKET_NAME).download_file(key, f'{path}test_img/{name}')

    image_path = f'{path}test_img/{name}'
    print()
    print('image_path', image_path)
    print()
    clothes_path = f'{path}test_clothes/{gender+category+item_id}.jpg'
    edge_path = f'{path}test_edge/{gender+category+item_id}.jpg'
    input_path = { 'image': image_path,'clothes': clothes_path ,'edge': edge_path}

    s3_path = 'model_image/' + user_key + str(top_id) + str(bottom_id)+'.jpg'
    print()
    print('s3_path', s3_path)
    print()
    location = s3.get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
    
    # 전처리되어 저장될 새 이미지의 url
    image_url = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/{s3_path}'

    # 만약에 이미 같은 이름의 이미지가 있으면 그걸 보여주기
    if check_model(image_url):
        print("already exist!")
        return image_url

    else:
        # 추론 함수 돌리기
        generated_image = inference_image(category, input_path, f'{path}results/{user_key+str(top_id)+str(bottom_id)}.jpg')
        generated_image = f'{path}results/{user_key+str(top_id)+str(bottom_id)}.jpg'
        
        # s3에 저장하기
        s3_resource.Bucket(BUCKET_NAME).upload_file(generated_image,s3_path)
        #로컬에 저장한 건 지우기
        if os.path.isfile(generated_image):
            os.remove(generated_image)



        # SQL에 새롭게 저장
        new_user = {
            'user_key': user_key,
            'model_url': image_url,
            'generate': True
        }

        new_user_id = insert_user(new_user)

        # 모델 이미지 자리에 새로 업로드받아 전처리된 이미지를 보여준다.
        #return render_template('model_image.html', image_name=image_url)
        return image_url


@app.route('/db')
def index():
    # new_user = request.json
    new_user = {
        'user_key': 'asdfszsdf2323asdxfxdfd',
        'model_image': 'c',
        'generate': False
    }
    new_user_id = insert_user(new_user)
    new_user = get_user(new_user_id)
    return jsonify(new_user)


# 업로드 HTML 렌더링
# @app.route('/upload')
# def render_file():
#    return render_template('upload.html')


# 파일 업로드 처리
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 업로드 받은 파일
        file = request.files['file']
        
        
        # 전처리 함수를 통과
        resized_img = resizing_human(file, model=net, temp_size=384)
        buf = io.BytesIO()
        resized_img.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        

        # 새 유저 키 생성
        user_key = get_user_key(32)
        print(user_key)

        top_id, bottom_id = 0,0
        # 저장할 storage 경로 + 파일명
        
        s3_path = 'model_image/' + user_key + str(top_id) + str(bottom_id)+'.jpg'
        s3.put_object(
            Bucket=BUCKET_NAME,
            Body=byte_im,
            Key=s3_path,
            ContentType=file.content_type
        )
        location = s3.get_bucket_location(Bucket=BUCKET_NAME)['LocationConstraint']
        
        # 전처리되어 저장된 새 이미지의 url
        image_url = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/{s3_path}'

        # SQL에 새롭게 저장
        new_user = {
            'user_key': user_key,
            'model_url': image_url,
            'generate': False
        }

        new_user_id = insert_user(new_user)

    # 모델 이미지 자리에 새로 업로드받아 전처리된 이미지를 보여준다.
    #return render_template('model_image.html', image_name=image_url)
    return image_url





# generate HTML 렌더링
# 상의 or 하의 item 선택 시 작동
# @app.route('/generate')
# def render_file():
# 1. 해당
# 이미지를 클릭할 시 해당 item 의 id 받아오기



if __name__ == '__main__':
    # 서버 실행
    app.run(debug=True, host='0.0.0.0', port=8001)
