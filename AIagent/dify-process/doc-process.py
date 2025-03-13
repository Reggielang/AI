from flask import Flask, request, jsonify
import os
import requests
import docx2txt
import subprocess
import tempfile

app = Flask(__name__)

def is_doc(file_path):
    """检查文件是否为.doc格式"""
    return file_path.lower().endswith('.doc')

def is_docx(file_path):
    """检查文件是否为.docx格式"""
    return file_path.lower().endswith('.docx')

def convert_doc_to_md(doc_path):
    """使用antiword将.doc文件转换为纯文本，然后尝试将其转换为markdown格式"""
    try:
        result = subprocess.run(['antiword', doc_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Failed to convert .doc file: " + result.stderr)
        return result.stdout
    except Exception as e:
        raise Exception("Error converting .doc to Markdown: " + str(e))

def convert_docx_to_md(docx_path):
    """使用docx2txt将.docx文件转换为纯文本，然后尝试将其转换为markdown格式"""
    try:
        content = docx2txt.process(docx_path)
        return content
    except Exception as e:
        raise Exception("Error converting .docx to Markdown: " + str(e))

@app.route('/convert', methods=['POST'])
def convert_file():
    data = request.get_json()

    if not data or 'file_path' not in data:
        return jsonify({'error': 'No file URL provided'}), 400

    file_url = 'http://' + data['file_id_address'] + data['file_path']
    app.logger.debug(f"Attempting to process file from URL: {file_url}")

    response = requests.get(file_url)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to download file'}), response.status_code

        # 获取文件扩展名并保留它
    _, file_extension = os.path.splitext(data['file_name'])
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(response.content)
        temp_path = temp_file.name

    try:
        app.logger.debug(f"File saved at {temp_path}")
        if is_doc(temp_path):
            markdown_content = convert_doc_to_md(temp_path)
        elif is_docx(temp_path):
            markdown_content = convert_docx_to_md(temp_path)
        else:
            app.logger.warning(f"Unsupported file type detected for file {temp_path}")
            os.unlink(temp_path)  # 清理临时文件
            return jsonify({'error': 'Unsupported file type'}), 400

        os.unlink(temp_path)  # 清理临时文件
        return jsonify({'markdown_content': markdown_content})
    except Exception as e:
        app.logger.error(f"Error processing file from URL {file_url}: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)