import os
import requests
import zipfile
import io

def collect_RAVDESS(data_path):
    urls = [
        "https://zenodo.org/records/1188976/files/Audio_Song_Actors_01-24.zip?download=1",
        "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
    ]

    os.makedirs(data_path, exist_ok=True)

    try:
        for url in urls:
            archive = requests.get(url).content
            with zipfile.ZipFile(io.BytesIO(archive)) as zip_ref:
                for file_info in zip_ref.infolist():
                    if not file_info.is_dir():
                        file_name = os.path.basename(file_info.filename)
                        target_path = os.path.join(data_path, file_name)
                        with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                            target.write(source.read())
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
