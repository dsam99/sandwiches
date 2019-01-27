import base64

def convert_and_save(b64_string, ext="png"):
    with open("tmp/imageToSave." + ext, "wb") as fh:
        fh.write(base64.b64decode(b64_string))


def img_to_base_64(img_filename):
    with open(img_filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string

base64_string = img_to_base_64('download.png')
print(base64.b64encode(base64_string))

# convert_and_save(base64_string, 'jpeg')
