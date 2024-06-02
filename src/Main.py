import argparse
from deepface import DeepFace
import easygui
import utils
import colorama
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

"""
    整体功能要使用命令行UI实现，在选择文件时使用系统UI
    整体集成DeepFace的功能，实现面部验证、面部查找、面部属性分析
"""


def verify_faces():
    """
    面部验证功能
    """
    print(colorama.Fore.YELLOW + "选择distances_metric参数: " + colorama.Style.RESET_ALL)
    print("1. euclidean")
    print("2. cosine")
    print("3. euclidean_l2")
    print("default: euclidean")

    choice = input(colorama.Fore.YELLOW + "请选择: " + colorama.Style.RESET_ALL)
    if choice == "1":
        metric = "euclidean"
    elif choice == "2":
        metric = "cosine"
    elif choice == "3":
        metric = "euclidean_l2"
    else:
        metric = "euclidean"
    img1_path = easygui.fileopenbox(msg="选择第一张图片，用于验证", title="选择图片")
    img2_path = easygui.fileopenbox(msg="选择第二张图片，用于验证", title="选择图片")

    if img1_path and img2_path:
        try:
            # __import__('ipdb').set_trace()
            result = DeepFace.verify(img1_path, img2_path, distance_metric=metric)
            print(colorama.Fore.GREEN + "验证结果: " + colorama.Style.RESET_ALL)
            utils.display_face_recognition_result(result, img1_path, img2_path)
        except Exception as e:
            print(colorama.Fore.RED + f"验证失败: {e}" + colorama.Style.RESET_ALL)
    else:
        easygui.msgbox("请选择两张图片进行验证")


def find_face_in_db():
    """
    在数据库中查找面部
    """
    img_path = easygui.fileopenbox(msg="选择图片", title="选择图片")
    db_path = easygui.diropenbox(msg="选择数据库文件夹", title="选择文件夹")
    print(img_path, db_path)

    if img_path and os.path.isdir(db_path):
        try:
            df = DeepFace.find(img_path=img_path, db_path=db_path)
            utils.display_find_in_db_result(df, img_path, db_path)
        except Exception as e:
            print(colorama.Fore.RED + f"查找失败: {e}" + colorama.Style.RESET_ALL)
    else:
        print(colorama.Fore.RED + "请选择图片和数据库文件夹" + colorama.Style.RESET_ALL)


def analyze_face():
    """
    面部属性分析功能
    """
    img_path = easygui.fileopenbox(msg="选择图片", title="选择图片")

    if img_path:
        try:
            obj = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'],
                                   enforce_detection=False)
            utils.display_analysis_result(obj, img_path)
        except Exception as e:
            print(colorama.Fore.RED + f"分析失败: {e}" + colorama.Style.RESET_ALL)
    else:
        print(colorama.Fore.RED + "请选择图片进行分析" + colorama.Style.RESET_ALL)


def real_time_face_recognition():
    """
    实时面部识别功能
    """
    face_db = easygui.diropenbox(msg="选择面部数据库文件夹", title="选择文件夹")
    if not os.path.isdir(face_db):
        print(colorama.Fore.RED + "请选择正确的文件夹" + colorama.Style.RESET_ALL)
        return

    DeepFace.stream(face_db)



if __name__ == '__main__':
    """ 
    Test process:
    using ./dataset as database, and ./testset for testing.
    1. verify_faces:    
        - using ./testset/img34.jpg and ./testset/img37.jpg. They are all Leonardo DiCaprio.
        - using ./testset/img33.jpg and ./testset/img34.jpg. They are not the same person.
    2. find_face_in_db:
         - using ./testset/img37.jpg and dataset.
    3. analyze_face:
        - using ./testset/img34.jpg.
        - Leonardo DiCaprio was born in 1974, and find image in https://www.esquire.com/uk/latest-news/a27251247/leonardo-dicaprio-could-play-a-con-man-in-guillermo-del-toros-nightmare-alley , in 2019
        - 2019 - 1974 = 45, close to 38
        - Leonardo DiCaprio has Italian and German ancestry (from https://en.wikipedia.org/wiki/Leonardo_DiCaprio, "Early life and acting background" Part), so he is white
        - Neutral and Man is obvious
    4. real_time_face_recognition:
        - you shuold build up your own image database with your own image
    """
    # 这里使用命令行来实现整体功能，不要使用easygui的GUI
    print(
        colorama.Fore.GREEN + "-----------------------------------------------------------------------------\n" + colorama.Style.RESET_ALL)
    print(colorama.Fore.GREEN + "欢迎使用面部识别系统\n" + colorama.Style.RESET_ALL)
    print(colorama.Fore.BLUE + "使用教程:\n" + colorama.Style.RESET_ALL)
    print("1. 面部验证功能")
    print("2. 在数据库中查找面部")
    print("3. 面部属性分析功能")
    print("4. 实时面部识别功能")
    print("5. 退出系统")
    print("6. 帮助菜单")
    print()

    while True:
        choice = input(colorama.Fore.YELLOW + "请选择功能: " + colorama.Style.RESET_ALL)
        if choice == "1":
            verify_faces()
        elif choice == "2":
            find_face_in_db()
        elif choice == "3":
            analyze_face()
        elif choice == "4":
            real_time_face_recognition()
        elif choice == "5":
            print(colorama.Fore.RED + "感谢使用，再见！" + colorama.Style.RESET_ALL)
            break
        elif choice == "6":
            print(colorama.Fore.BLUE + "使用教程:\n" + colorama.Style.RESET_ALL)
            print("1. 面部验证功能")
            print("2. 在数据库中查找面部")
            print("3. 面部属性分析功能")
            print("4. 实时面部识别功能")
            print("5. 退出系统")
            print("6. 帮助菜单")
        else:
            print("无效的选择，请重新选择, 输入5查看帮助菜单")
            continue

