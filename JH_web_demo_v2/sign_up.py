from flask      import Flask, request, jsonify, current_app
from flask.json import JSONEncoder
from sqlalchemy import create_engine, text

def get_user(user_id):
    user = current_app.database.execute(text("""
        SELECT 
            user_id,
            model_url,
            generate
        FROM User
        WHERE user_id = :user_id
    """), {
        'user_id' : user_id
    }).fetchone()

    return {
        'user_id'      : user['user_id'],
        'model_url'   : user['model_url'],
        'generate' : user['generate']
    } if user else None

# HTTP 요청을 통해 전달받은 회원가입 정보를 데이터 베이스에 저장함
def insert_user(user):
    return current_app.database.execute(text("""
        INSERT INTO User (
            model_url,
            generate
        ) VALUES (
            :model_url,
            :generate
        )
    """), user)#.lastrowid#.inserted_primary_key # 새로 사용자가 생성되면 새로 생성된 사용자의 아이디를 읽어들인다.


def insert_item(item):
    current_app.database.execute(text("""
        INSERT INTO Item (
            item_number,
            item_url,
            category,
            sex
        ) VALUES (
            :item_number,
            :item_url,
            :category,
            :sex
        )
        """), item)

def insert_mask(mask):
    current_app.database.execute(text("""
        INSERT INTO Mask (
            item_number,
            mask_url
        ) VALUES (
            :item_number,
            :mask_url
        )
        """), mask)


def insert_generate(generate):
    return current_app.database.execute(text("""
        INSERT INTO Generate(
            model_url,
            item_number,
            generate_url,
            ) VALUES (
                :model_url,
                :item_number
                :generate_url
            )
        """), generate)


def check_model(model_url):
    image = current_app.database.execute(text("""
        SELECT
            model_url
        FROM User
        WHERE model_url =:model_url
        """), {
            'model_url' :model_url
        }).fetchone()

    return True if image else None


def check_category(item_url):
    item = current_app.database.execute(text("""
        SELECT
            category
        FROM Item
        WHERE item_url =:item_url
        """), {
            'item_url' :item_url
        }).fetchone()
    return item['category']

def get_item_num(item_url):
    item = current_app.database.execute(text("""
        SELECT
            item_number
        FROM Item
        WHERE item_url =:item_url
        """), {
            'item_url' :item_url
        }).fetchone()
    return item['item_number']

def get_mask_url(item_num):
    mask = current_app.database.execute(text("""
        SELECT
            mask_url
        FROM Mask
        WHERE item_num =:item_num
        """), {
            'item_num' :item_num
        }).fetchone()
    return mask['mask_url']

def get_items(category, sex):
    items = current_app.database.execute(text("""
        SELECT *
        FROM Item
        WHERE category =:category and sex =:sex
        """), {
            'category' :category,
            'sex' :sex
        }).fetchall()
    return items
