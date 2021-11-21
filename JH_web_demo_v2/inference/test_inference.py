from inference import inference_image
    

if __name__ == '__main__':
<<<<<<< Updated upstream
    image_path = './dataset/test_img/female_model.jpg'
    clothes_path = './dataset/test_clothes/0988551_02.jpeg'
    edge_path = './dataset/test_edge/0988551_02.jpeg'
    input_path = { 'image': image_path,'clothes': clothes_path ,'edge': edge_path}
    result_path = inference_image(input_path , "./result.jpg")
=======
    image_path = './dataset/test_img/7n5174dfwm0c51s606zpd97t9p5e46yb0000.jpg'
    clothes_path = './dataset/test_clothes/mbottomitem3.jpg'
    edge_path = './dataset/test_edge/mbottomitem3.jpg'
    input_path = { 'image': image_path,'clothes': clothes_path ,'edge': edge_path}
    inference_image('bottom', input_path , "./result.jpg")
>>>>>>> Stashed changes
