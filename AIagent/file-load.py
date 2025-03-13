from flask import Flask, request, jsonify
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB


@app.route('/upload', methods=['POST'])
def upload_file():
    print("Headers: ", request.headers)
    # 检查是否有文件在请求中
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']

    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 打印文件信息
    print(f"Received file: {file.filename}")
    print(f"Content type: {file.content_type}")

    print(f"File size: {len(file.read())} bytes")
    # 确保存储目录存在
    upload_directory = 'uploads'
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)

    # 构建文件保存路径
    filepath = os.path.join(upload_directory, file.filename)
    file.save(filepath)
    return jsonify({"message": "File uploaded successfully"}), 200


if __name__ == '__main__':
    app.run(port=5000, debug=True)