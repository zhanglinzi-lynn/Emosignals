# Generation Recipe: Detecting Multi-Modal Fake AIGC Information from the Perspective of Emotional Manipulation Purpose
## Introduction
The implementation of Emosignals, a creative multi-modal fake AIGC detection model based on emotional signals and emotional fluctuation.



## Dataset

- **Data Format**:
  ```
    {
        "video_id":"7299305894641208607",
        "description":"putin and Kim Jung are either both socislly ackward or hoth have trust issues! #funnypolitics #trump2024 #politicalhumor ",
        "annotation":"fake",
        "user_certify":0,  # 1 if the account is verified else 0
        "user_description":"Your business, relationship and life coach!",
        "publish_time":1699502099000,
        "event":"trust issues Kim Putin"
    }
  ```


## Quick Start
You can train Emosignals by following code:
 ```
 
  python main.py  

 

