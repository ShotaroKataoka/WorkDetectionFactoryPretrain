# work_detection_factory_pretrain
Improving work detection by segmentation heuristics pre-training on factory operations video
[This](https://drive.google.com/drive/folders/1VNGoTJCoog4QfW-So60D-mCXslkOW9SD?usp=sharing) is an example of the result. Due to runtime random numbers, there is no guarantee that the results will be exactly the same.

# How to start
1. Clone this project
2. [Download](https://drive.google.com/drive/folders/1KvXQ5CzhU173uSxVAtkNO0yABsatpmFp?usp=sharing) data, and unzip.
3. Set configure `params/predict_class.yaml`.
4. `python train.py`
5. `python predict.py`
5. `python test_score.py`

