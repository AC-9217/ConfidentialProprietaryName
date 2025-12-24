# 生产测试样例
# 用于生成测试用的图片文件
import fitz
from PIL import Image
import os


def create_image_red():
    """
    Create a red test image.
    创建一个红色的测试图片。
    """
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save("./pic/test_image_1.jpg")
    print("Created test_image_1.jpg")
def create_image_blue():
    """
    Create a blue test image.
    创建一个蓝色的测试图片。
    """
    img = Image.new('RGB', (100, 100), color = 'blue')
    img.save("./pic/test_image_2.jpg")
    print("Created test_image_2.jpg")
def create_image_green():
    """
    Create a green test image.
    创建一个绿色的测试图片。
    """
    img = Image.new('RGB', (100, 100), color = 'green')
    img.save("./pic/test_image_3.jpg")
    print("Created test_image_3.jpg")
def create_image_yellow():
    """
    Create a yellow test image.
    创建一个黄色的测试图片。
    """
    img = Image.new('RGB', (100, 100), color = 'yellow')
    img.save("./pic/test_image_4.jpg")
    print("Created test_image_4.jpg")

if __name__ == "__main__":
    create_image_red()
    create_image_blue()
    create_image_green()
    create_image_yellow()   
