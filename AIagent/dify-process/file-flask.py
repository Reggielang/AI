from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)


# 上述提到的代码逻辑封装成函数
def process_pdf(pdf_file):
    # 假设这里是你提供的PDF处理逻辑
    pass


@app.route('/process-pdf', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        filepath = os.path.join("uploads", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)

        # 调用你的PDF处理逻辑
        result = process_pdf(filepath)

        # 返回处理结果或文件下载链接
        response = {
            "message": "File processed successfully",
            "download_url": f"/downloads/{filename.split('.')[0]}_model.pdf"
        }
        return jsonify(response), 200


@app.route('/downloads/<path:filename>', methods=['GET'])
def download_file(filename):
    uploads = os.path.join(os.getcwd(), "output")
    return send_from_directory(directory=uploads, path=filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)