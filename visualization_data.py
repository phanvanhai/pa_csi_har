import matplotlib.pyplot as plt 
import numpy as np

# [SỬA ĐỔI]: Đã xóa các import rác không được sử dụng trong mã (load_iris, RandomForestClassifier, train_test_split) 
# để giảm thiểu dung lượng bộ nhớ.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import os # [SỬA ĐỔI]: Import thêm os (nếu sau này cần quản lý thư mục lưu ảnh)

def draw_acc(epochs, train, val, save_path="accuracy_plot.png"):
    xpoints = np.arange(0, epochs, dtype=int)
    train = np.array(train)
    val = np.array(val)
    
    plt.figure() # [SỬA ĐỔI]: Luôn khởi tạo figure mới để tránh vẽ đè hình cũ
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(xpoints, train)
    plt.plot(xpoints, val)
    plt.legend(['train', 'val'], loc='lower right')
    
    # [SỬA ĐỔI]: Lưu thành file ảnh thay vì show pop-up để không làm treo Kaggle
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

def draw_loss(history, save_path="loss_plot.png"):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    
    # [SỬA ĐỔI]: Lưu file ảnh
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def draw_confusion_matrix(y_test, y_pred, save_path="confusion_matrix_basic.png"):
    plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    target_names=['No movement', 'Falling', 'Sitting down/standing up', 'Walking', 'Turning', 'Picking up']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    
    # [SỬA ĐỔI]: Lưu file ảnh
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def draw_confusion_matrix_2(y_true, y_pred, save_path="confusion_matrix_heatmap.png"):
    # Define the mapping from numeric labels to string labels
    label_mapping = {
        0: 'No movement',
        1: 'Falling',
        2: 'Sitting down / Standing up',
        3: 'Walking',
        4: 'Turning',
        5: 'Picking up a pen'
    }

    # Convert the numeric labels to string labels for display purposes
    labels = [label_mapping[i] for i in sorted(label_mapping.keys())]

    # Generate confusion matrix with numeric values
    cm = confusion_matrix(y_true, y_pred, labels=sorted(label_mapping.keys()))

    # Convert confusion matrix to percentage format (row-wise)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Set up the figure and axes
    plt.figure(figsize=(10, 8)) # [SỬA ĐỔI]: Tăng một chút size để chữ hiển thị trên Kaggle không bị cắt

    # Plot the heatmap with string labels
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)

    # Add axis labels and title
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Reference', fontsize=12)
    plt.title('Confusion Matrix of MultiEnv LOS (Office)', fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right') # [SỬA ĐỔI]: Thêm ha='right' để nhãn trục x xoay đẹp hơn

    # Show the plot
    plt.tight_layout()
    
    # [SỬA ĐỔI]: Đổi plt.show() thành plt.savefig() với độ phân giải cao (dpi=300) để bạn dễ dàng chèn vào báo cáo
    plt.savefig(save_path, dpi=300)
    plt.close()