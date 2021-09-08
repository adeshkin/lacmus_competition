# 1st Lacmus Computer Vision Competition: 6th place solution

#### TASK - to detect lost people in aerial images
#### 2-d detection:
![header](images/example.png)

- [1st Lacmus Computer Vision Competition](https://ods.ai/competitions/lacmus-cvc-soc2021)


## Dataset

Download
[dataset](https://ods.ai/competitions/lacmus-cvc-soc2021/data)

Data analysis:
- `EDA.ipynb`
- `apply_augmenatations.ipynb`

## Solution:

* Fasterrcnn

## Environment:
```bash
git clone https://github.com/adeshkin/lacmus_competition
cd lacmus_competition
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```


## Training

```bash
python main.py 
```

Log example:
![header](/home/cds-k/Desktop/lacmus_competition/images/graphic_val_ap_iou_0_5.png)