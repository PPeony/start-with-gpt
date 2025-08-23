import os

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ocr.v20181119 import ocr_client, models


def recognize_text(image_path):
    try:
        # 实例化认证对象
        cred = credential.Credential(os.getenv("TENCENT_API_SECRET_ID"), os.getenv("TENCENT_API_SECRET_KEY"))

        # 实例化http选项
        httpProfile = HttpProfile()
        httpProfile.endpoint = "ocr.tencentcloudapi.com"

        # 实例化client选项
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile

        # 实例化OCR client对象
        client = ocr_client.OcrClient(cred, "ap-guangzhou", clientProfile)

        # 读取图片文件并转为Base64
        with open(image_path, 'rb') as f:
            image = f.read()
        import base64
        image_base64 = base64.b64encode(image).decode()

        # 实例化请求对象
        req = models.GeneralBasicOCRRequest()
        req.ImageBase64 = image_base64

        # 发起OCR识别请求
        resp = client.GeneralBasicOCR(req)

        # 提取识别结果
        result = []
        for text_item in resp.TextDetections:
            result.append(text_item.DetectedText)

        return '\n'.join(result)

    except Exception as e:
        print(f"OCR识别发生错误: {str(e)}")
        return None
if __name__ == "__main__":
    image_path = "试题1.png"
    result = recognize_text(image_path)
    print(result)