
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Itesm(db.Model):
    __tablename__ = "items"
    item_id = db.Column(db.Integer, primary_key = True)
    item = db.Column(db.String, unique=True, nullable=False)
    category = db.Column(db.String, nullable=False)
    sex = db.Column(db.String, nullable=False)

class User(db.Model):
    __tablename__ = "user"
    user_id = db.Column(db.Integer, primary_key = True)
    user_key = db.Column(db.Integer, nullable=False)
    model_image = db.Column(db.String, , unique=True, default='~~~.jpg')
    generate = db.Column(db.Boolean, unique=False, default = False)

class Generate(db.Model):
    __tablename__ = 'generate'
    generate_id = db.Column(db.Integer, primary_key = True)
    item = db.Column(db.String, db.ForeignKey('items.item'), nullable=True)
    model_image = db.Column(db.String, db.ForeignKey('user.model_image'), nullable=False)
    generate_image = db.Column(db.String, nullable=False)