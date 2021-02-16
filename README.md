# OCR Rest-API with Django rest framework

### Framework Used
- [Tensorflow](https://www.tensorflow.org/)
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)
- [Django Rest Framework](https://www.django-rest-framework.org)


### Features

- Ocr KTP (Indonesian Identity)


### Configuration

```bash
$ # Virtualenv modules installation (Unix based systems)
$ virtualenv --no-site-packages env
$ source env/bin/activate
$
$ # Virtualenv modules installation (Windows based systems)
$ # virtualenv --no-site-packages env
$ # .\env\Scripts\activate
$ 
$ # Install modules
$ # SQLIte version
$ pip3 install -r requirements.txt
$
$ # Create tables
$ python3 manage.py migrate
$ # Create a superuser
$ python3 manage.py createsuperuser
$ # Generate token
$ python3 manage.py drf_create_token registered_username
```
<br />

## API Endpoint
- http://host/api/ocr/model_id?format=json (method: POST)
    - api for the ocr process
    - params:
        - file: file to ocr
    - header:
        - Authorization: Token <generated_token> 
    - response:
        ```bash
        {
            "filename": "uploaded-file.png",
            "width": 1005,
            "height": 623,
            "data": {
                "province": [
                    {
                        "ymin": 3.980022942647338,
                        "xmin": 309.973214417696,
                        "ymax": 40.45701538771391,
                        "xmax": 705.5190673470497,
                        "text": "PROVINSI JAWA BARAT\n\f"
                    }
                ],
                "city": [
                    {
                        "ymin": 37.83664844185114,
                        "xmin": 381.3558992743492,
                        "ymax": 72.47854132205248,
                        "xmax": 625.4671186208725,
                        "text": "KOTA BEKASI\n\f"
                    }
                ],
                "NIK": [
                    {
                        "ymin": 80.39181964099407,
                        "xmin": 216.19754947721958,
                        "ymax": 135.65755248069763,
                        "xmax": 656.582301557064,
                        "text": "1234567891618636\n\f"
                    }
                ],
                "Name": [
                    {
                        "ymin": 142.9527358263731,
                        "xmin": 22.621495658531785,
                        "ymax": 175.86385509371758,
                        "xmax": 489.2742995917797,
                        "text": "Nama            . LUGMAN SUNGKAR\n\f"
                    }
                ],
                "dob": [
                    {
                        "ymin": 171.78611135482788,
                        "xmin": 18.827045392245054,
                        "ymax": 207.74667739868164,
                        "xmax": 612.0886364579201,
                        "text": "Tempat/Tgi Lahir : TANJUNG KARANG. 19-05-1972\n\f"
                    }
                ],
                "address": [
                    {
                        "ymin": 230.64498394727707,
                        "xmin": 15.393000463955104,
                        "ymax": 293.8825688660145,
                        "xmax": 624.9208661913872,
                        "text": "Alamat             . TAMAN TYTYAN INDAH BLOK J4\nNO.11\n\f"
                    }
                ],
                "rtrw": [
                    {
                        "ymin": 293.48761489987373,
                        "xmin": 74.6471556276083,
                        "ymax": 323.66689187288284,
                        "xmax": 346.56164422631264,
                        "text": "RTRW     : 006 : 010\n\f"
                    }
                ],
                "subdistrict": [
                    {
                        "ymin": 320.60971200466156,
                        "xmin": 70.66152412444353,
                        "ymax": 352.83020955324173,
                        "xmax": 375.6223453581333,
                        "text": "KelDesa     . KALi BARU\n\f"
                    }
                ],
                "district": [
                    {
                        "ymin": 347.0303350687027,
                        "xmin": 73.15130364149809,
                        "ymax": 385.98476803302765,
                        "xmax": 426.0538324713707,
                        "text": "Kecamatan : MEDAN SATRIA\n\f"
                    }
                ],
                "exp": [
                    {
                        "ymin": 496.2964579463005,
                        "xmin": 17.907827720046043,
                        "ymax": 533.9860059022903,
                        "xmax": 369.63953644037247,
                        "text": "Berlaku Hingga :12-08-2017\n\f"
                    }
                ]
            }
        }
        ```
- http://host/api/models?format=json (method: GET)
    - api to get the model_id list
    - header:
        - Authorization: Token <generated_token> 
    - response:
        ```bash
        {
            "total_model": 1,
            "model_id": [
                "ktp"
            ]
        }
        ```

## Models (Tensorflow model)
you can add new model at /models/model_id and replace the label map name with `label_map.pbtxt`