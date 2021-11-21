import string 
import random 

def get_user_key(key_len): 
    string_lower = string.ascii_lowercase#영어 소문자 
    string_upper = string.ascii_uppercase #영어 대문자 
    string_digits = string.digits #숫자 
    
    #영어 소문자와 숫자만 활용해서 키를 만든다. 
    key = "" 
    for i in range(key_len) :
        ran = random.randint(0, 1) 
        if ran == 0: 
            key = key + random.choice(string_lower) 
        else: 
            key = key + random.choice(string_digits) 

   # if key in user_list:
   #     while key not in user_list:
   #         key = get_user_key(key_len, user_list)
    return key
    

