import os
import glob

def clean_windows_paths():
    """
    Quét toàn bộ thư mục hiện tại, tìm các tệp .py và tự động chuyển đổi
    chuẩn đường dẫn từ Windows (\\) sang chuẩn POSIX (/) để tương thích với Kaggle.
    """
    print("Bắt đầu quét và chuẩn hóa đường dẫn cho môi trường Linux/Kaggle...")
    
    # Tìm tất cả các tệp Python trong thư mục và thư mục con
    py_files = glob.glob("**/*.py", recursive=True)
    fixed_files_count = 0
    
    # Tên của chính file này để tránh tự sửa đổi chính nó
    current_script_name = os.path.basename(__file__)

    for file_path in py_files:
        if os.path.basename(file_path) == current_script_name:
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            # Thay thế double-backslash thành single-slash
            modified_content = content.replace('\\\\', '/')
            
            if content != modified_content:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(modified_content)
                print(f" [+] Đã sửa đường dẫn trong tệp: {file_path}")
                fixed_files_count += 1
        except Exception as e:
            print(f" [!] Lỗi khi xử lý tệp {file_path}: {e}")

    print(f"\nHoàn tất! Đã chuẩn hóa thành công {fixed_files_count} tệp.")
    print("Dự án đã sẵn sàng để commit và push lên GitHub.")

if __name__ == "__main__":
    clean_windows_paths()