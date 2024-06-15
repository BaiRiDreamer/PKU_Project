import os
import easygui
import matplotlib.pyplot as plt

def display_face_recognition_result(result, img1_path, img2_path):
    """
    显示面部识别结果,
    result:{'verified': True, 'distance': -2.220446049250313e-16, 'threshold': 0.68, 'model': 'VGG-Face', 'detector_backend': 'opencv', 'similarity_metric': 'cosine', 'facial_areas': {'img1': {'x': 196, 'y': 234, 'w': 625, 'h': 625, 'left_eye': (633, 483), 'right_eye': (367, 483)}, 'img2': {'x': 196, 'y': 234, 'w': 625, 'h': 625, 'left_eye': (633, 483), 'right_eye': (367, 483)}}, 'time': 1.18}
    """
    # 使用matplotlib显示图片，将两张图片显示在一起，方便对比，使用result中的facial_areas信息，标记出面部区域
    # 除此之外，还可以标记出眼睛的位置
    # 最后要显示面部识别结果，显示result中的verified字段, 如果为True, 则显示为验证成功，否则显示为验证失败
    # 另外显示result中的distance字段，表示两张图片的相似度,显示threshold字段，表示阈值
    # 直接将result中的信息显示在图片下方即可,不要print出来
    img1 = plt.imread(img1_path)
    img2 = plt.imread(img2_path)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis("off")
    plt.title("Image 1")
    plt.gca().add_patch(plt.Rectangle((result["facial_areas"]["img1"]["x"], result["facial_areas"]["img1"]["y"]),
                                        result["facial_areas"]["img1"]["w"], result["facial_areas"]["img1"]["h"],
                                        edgecolor="r", facecolor="none"))
    plt.scatter(result["facial_areas"]["img1"]["left_eye"][0], result["facial_areas"]["img1"]["left_eye"][1], c="r", s=10)
    plt.scatter(result["facial_areas"]["img1"]["right_eye"][0], result["facial_areas"]["img1"]["right_eye"][1], c="r", s=10)

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis("off")
    plt.title("Image 2")
    plt.gca().add_patch(plt.Rectangle((result["facial_areas"]["img2"]["x"], result["facial_areas"]["img2"]["y"]),
                                        result["facial_areas"]["img2"]["w"], result["facial_areas"]["img2"]["h"],
                                        edgecolor="r", facecolor="none"))

    plt.scatter(result["facial_areas"]["img2"]["left_eye"][0], result["facial_areas"]["img2"]["left_eye"][1], c="r", s=10)
    plt.scatter(result["facial_areas"]["img2"]["right_eye"][0], result["facial_areas"]["img2"]["right_eye"][1], c="r", s=10)

    plt.suptitle("Face Recognition Result")
    plt.figtext(0.5, 0.05, f"Verified: {result['verified']}, Distance: {result['distance']}, Threshold: {result['threshold']}\n Model: {result['model']}, Detector Backend: {result['detector_backend']}\n Similarity Metric: {result['similarity_metric']}, Time: {result['time']}", ha="center")

    plt.show()


# results (List[pd.DataFrame]): A list of pandas dataframes. Each dataframe corresponds
# to the identity information for an individual detected in the source image.
# The DataFrame columns include:
#
# - 'identity': Identity label of the detected individual.
#
# - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the
# target face in the database.
#
# - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the
# detected face in the source image.
#
# - 'threshold': threshold to determine a pair whether same person or different persons
#
# - 'distance': Similarity score between the faces based on the
# specified model and distance metric
def display_find_in_db_result(df_list, img_path, db_path):
    """
    显示在数据库中查找面部的结果, 即首先显示源图片，之后将其他的所有图片也显示出来(6个一组)
    """
    df_list_len = len(df_list)
    if df_list_len == 0:
        img = plt.imread(img_path)
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Source Image")
        plt.figtext(0.5, 0.05, "No face detected in the database", ha="center")
        plt.show()


    else:
        for i, df in enumerate(df_list):
            if i % 6 == 0:
                plt.figure(figsize=(15, 10))
                img = plt.imread(img_path)
                plt.subplot(2, 3, 1)
                plt.imshow(img)
                plt.axis("off")
                plt.gca().add_patch(plt.Rectangle((df["source_x"].iloc[0], df["source_y"].iloc[0]),
                                                    df["source_w"].iloc[0], df["source_h"].iloc[0],
                                                    edgecolor="r", facecolor="none"))

                plt.title("Source Image")

            img = plt.imread(os.path.join(df['identity'].iloc[0]))
            plt.subplot(2, 3, i % 6 + 2)
            plt.imshow(img)
            plt.axis("off")
            plt.gca().add_patch(plt.Rectangle((df["target_x"].iloc[0], df["target_y"].iloc[0]),
                                                df["target_w"].iloc[0], df["target_h"].iloc[0],
                                                edgecolor="r", facecolor="none"))
            plt.title(f"Image {i + 1}")

            if i % 6 == 5 or i == df_list_len - 1:
                plt.suptitle("Find Face in Database Result")
                plt.figtext(0.5, 0.05, f"Source Image: {img_path}, Database: {db_path}", ha="center")
                plt.show()


def display_analysis_result(result, img_path):
    """
    显示面部分析结果的图表
    """
    result = result[0]
    print(result)
    img = plt.imread(img_path)
    plt.figure(figsize=(10, 4))

    plt.imshow(img)
    plt.axis("off")
    plt.title("Face Image")

    plt.gca().add_patch(plt.Rectangle((result["region"]["x"], result["region"]["y"]),
                                        result["region"]["w"], result["region"]["h"],
                                        edgecolor="r", facecolor="none"))

    plt.figtext(0.5, 0.95, f"Age: {result['age']}", ha="center")
    plt.figtext(0.5, 0.9, f"Race: {result['dominant_race']}", ha="center")
    plt.figtext(0.5, 0.85,f'Emotion: {result["dominant_emotion"]}', ha="center")
    plt.figtext(0.5, 0.8,f'Gender: {result["dominant_gender"]}', ha="center")
    plt.show()




def batch_process_images():
    """
    批量处理文件夹中的图片
    """
    folder_path = filedialog.askdirectory(title="选择批量处理文件夹")

    if folder_path:
        progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        progress.pack(pady=20)

        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        total_images = len(images)

        for i, filename in enumerate(images):
            img_path = os.path.join(folder_path, filename)
            analyze_face(img_path)  # 对每张图片进行分析
            progress['value'] = ((i + 1) / total_images) * 100
            root.update_idletasks()

        progress.pack_forget()
        messagebox.showinfo("完成", "批量处理完成")
    else:
        messagebox.showwarning("警告", "请选择文件夹进行批量处理")


def analyze_images(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            path = os.path.join(directory, filename)
            result = DeepFace.analyze(img_path=path, actions=['age', 'gender', 'race', 'emotion'])
            results[filename] = result
    return results